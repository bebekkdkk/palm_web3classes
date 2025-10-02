import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from db_connection import DatabaseConnection

APP_TZ = os.environ.get('APP_TZ', 'Asia/Jakarta')

def now_local():
    try:
        return datetime.now(ZoneInfo(APP_TZ))
    except Exception:
        return datetime.now()

class ValidationHandler:
    def __init__(self):
        self.db = DatabaseConnection()
        self.ticket_table = 'ticket'
        self.validations_table = 'validations'

    def load_database(self):
        """Load detection results from ticket table and map to legacy shape used by app."""
        try:
            query = f"""
                SELECT id, file_name, group_id, class_result, status, place, username, created_at, updated_at,
                       total_process_time, detection_time, classification_time
                FROM {self.ticket_table}
                ORDER BY created_at DESC
            """
            results = self.db.execute_query(query, fetch=True)
            
            if results:
                normalized = []
                for result in results:
                    raw_place = result.get('place')
                    place_val = raw_place
                    # Normalize place: '-' stays '-', numeric strings -> int, 0 stays 0
                    if isinstance(raw_place, str):
                        p = raw_place.strip()
                        if p == '-':
                            place_val = '-'
                        else:
                            try:
                                place_val = int(p)
                            except Exception:
                                place_val = p
                    elif isinstance(raw_place, (int, float)):
                        place_val = int(raw_place)
                    normalized.append({
                        'id': result['id'],
                        'file_name': result['file_name'],
                        'group': result.get('group_id'),
                        'class_result': result['class_result'],
                        'status': result['status'],
                        'place': place_val,
                        'username': result['username'],
                        'created_at': result['created_at'].strftime('%Y-%m-%d %H:%M:%S') if result.get('created_at') else None,
                        'total_process_time': result.get('total_process_time', 0.0),
                        'detection_time': result.get('detection_time', 0.0),
                        'classification_time': result.get('classification_time', 0.0)
                    })
                return normalized
            
            return []
            
        except Exception as e:
            print(f"Error loading database: {str(e)}")
            return []

    def save_database(self, data):
        """Bulk insert detection results into ticket table."""
        try:
            # This method now adds new records to database instead of overwriting
            for item in data:
                if isinstance(item, dict):
                    query = f"""
                        INSERT INTO {self.ticket_table}
                        (file_name, group_id, class_result, status, place, username, created_at, updated_at, 
                         total_process_time, detection_time, classification_time)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    # Convert timestamp string to datetime if needed
                    created_at = item.get('created_at', now_local())
                    if isinstance(created_at, str):
                        try:
                            created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
                        except:
                            created_at = now_local()
                    updated_at = now_local()
                    
                    params = (
                        item.get('file_name'),
                        item.get('group'),
                        item.get('class_result'),
                        item.get('status', 'open'),
                        item.get('place'),
                        item.get('username'),
                        created_at,
                        updated_at,
                        item.get('total_process_time', 0.0),  # Total waktu dari awal sampai selesai
                        item.get('detection_time', 0.0),     # Waktu untuk proses deteksi
                        item.get('classification_time', 0.0) # Waktu untuk proses klasifikasi
                    )
                    
                    self.db.execute_query(query, params)
            
            return True
            
        except Exception as e:
            print(f"Error saving to database: {str(e)}")
            return False

    def add_detection_result(self, file_name, group, class_result, status, place, username, 
                           total_process_time=0.0, detection_time=0.0, classification_time=0.0):
        """Add a single detection result to ticket table with timing information."""
        try:
            query = f"""
                INSERT INTO {self.ticket_table}
                (file_name, group_id, class_result, status, place, username, created_at, updated_at,
                 total_process_time, detection_time, classification_time)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            ts = now_local()
            # Store place as text to support '-'
            place_db = place if isinstance(place, str) else str(place)
            params = (file_name, group, class_result, status, place_db, username, ts, ts,
                     total_process_time, detection_time, classification_time)
            self.db.execute_query(query, params)
            return True
            
        except Exception as e:
            print(f"Error adding detection result: {str(e)}")
            return False

    def add_timing_info(self, ticket_id, total_process_time=0.0, detection_time=0.0, classification_time=0.0):
        """Update timing information for an existing ticket record."""
        try:
            query = f"""
                UPDATE {self.ticket_table}
                SET total_process_time = %s, detection_time = %s, classification_time = %s, updated_at = %s
                WHERE id = %s
            """
            params = (total_process_time, detection_time, classification_time, now_local(), ticket_id)
            self.db.execute_query(query, params)
            return True
            
        except Exception as e:
            print(f"Error updating timing info: {str(e)}")
            return False

    def get_timing_statistics(self, group_id=None, date_filter=None):
        """Get timing statistics for performance analysis."""
        try:
            base_query = f"""
                SELECT 
                    AVG(total_process_time) as avg_total_time,
                    AVG(detection_time) as avg_detection_time,
                    AVG(classification_time) as avg_classification_time,
                    MIN(total_process_time) as min_total_time,
                    MAX(total_process_time) as max_total_time,
                    COUNT(*) as total_records
                FROM {self.ticket_table}
                WHERE total_process_time > 0
            """
            
            params = []
            if group_id:
                base_query += " AND group_id = %s"
                params.append(group_id)
            
            if date_filter:
                base_query += " AND DATE(created_at) = %s"
                params.append(date_filter)
            
            result = self.db.execute_query(base_query, params, fetchone=True)
            
            if result:
                return {
                    'avg_total_time': float(result.get('avg_total_time', 0) or 0),
                    'avg_detection_time': float(result.get('avg_detection_time', 0) or 0),
                    'avg_classification_time': float(result.get('avg_classification_time', 0) or 0),
                    'min_total_time': float(result.get('min_total_time', 0) or 0),
                    'max_total_time': float(result.get('max_total_time', 0) or 0),
                    'total_records': int(result.get('total_records', 0) or 0)
                }
            return None
            
        except Exception as e:
            print(f"Error getting timing statistics: {str(e)}")
            return None

    def update_valid_status(self, image_id, place):
        """Toggle status open/close for a specific crop in ticket table.
        Returns (success, message).
        """
        try:
            # Update the status in ticket table
            query = f"""
                UPDATE {self.ticket_table}
                SET status = CASE 
                    WHEN status = 'open' THEN 'close' 
                    ELSE 'open' 
                END,
                    updated_at = %s
                WHERE file_name LIKE %s AND place = %s
            """
            
            # Extract base filename from image_id
            base_filename = image_id.replace('(-)', '').replace('(0)', '')
            if '(' not in base_filename:
                base_filename = base_filename + '%'

            rowcount = self.db.execute_query(query, (now_local(), f"{base_filename}%", place))
            if rowcount and rowcount > 0:
                msg = f"Updated status for image {image_id} place {place}"
                print(msg)
                return True, msg
            else:
                msg = f"No records found for image {image_id} place {place}"
                print(msg)
                return False, msg
                
        except Exception as e:
            print(f"Error in update_valid_status: {str(e)}")
            return False, str(e)

    def set_status_for_base(self, base_name, group, new_status, uploader_username):
        """Set status for all records of a base_name within a group if uploader matches.
        Returns (success, message).
        """
        try:
            # Find the uploader (place '-')
            find_query = f"""
                SELECT username FROM {self.ticket_table}
                WHERE file_name LIKE %s AND (place = '-' OR place = %s) AND group_id = %s
                ORDER BY created_at ASC LIMIT 1
            """
            # Try both '-' and string '-'
            uploader_row = self.db.execute_query(find_query, (f"{base_name}%", '-', group), fetchone=True)
            if not uploader_row:
                uploader_row = self.db.execute_query(find_query, (f"{base_name}%", '-1', group), fetchone=True)
            if not uploader_row:
                return False, 'Original entry not found'
            if uploader_row['username'] != uploader_username:
                return False, 'Tidak bisa mengubah, hanya uploader yang boleh.'

            update_query = f"""
                UPDATE {self.ticket_table}
                SET status = %s, updated_at = %s
                WHERE file_name LIKE %s AND group_id = %s
            """
            rc = self.db.execute_query(update_query, (new_status, now_local(), f"{base_name}%", group))
            if rc and rc > 0:
                return True, 'Updated'
            return False, 'No records updated'
        except Exception as e:
            return False, str(e)

    def verify_classification(self, crop_path, verified_label, username, group):
        """Verify/update classification and save to validations table. Returns (bool, message)."""
        try:
            # Extract filename from crop_path
            filename = os.path.basename(crop_path)
            if filename.startswith('/'):
                filename = filename[1:]
            if filename.startswith('static/uploads/'):
                filename = filename[15:]
            
            # Extract place from filename
            place = None
            if '(' in filename and ')' in filename:
                try:
                    place_str = filename.split('(')[1].split(')')[0]
                    if place_str.isdigit():
                        place = int(place_str)
                except:
                    pass
            
            if place is None:
                msg = f"Could not extract place from filename: {filename}"
                print(msg)
                return False, msg
            
            # Save verification to validations table
            ok = self._save_to_validations_table(filename, group, verified_label, place, username, True)
            if ok:
                return True, 'Saved'
            return False, 'Failed to save'
                
        except Exception as e:
            print(f"Error in verify_classification: {str(e)}")
            return False, str(e)

    def _save_to_validations_table(self, filename, group, new_class, place, username, valid_status):
        """Save validation record to validations table"""
        try:
            query = f"""
                INSERT INTO {self.validations_table}
                (file_name, user_group, class_result, place, username_yang_mengubah, timestamp, valid_status)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            params = (filename, group, new_class, place, username, now_local(), valid_status)
            
            result = self.db.execute_query(query, params)
            
            if result:
                print(f"Saved validation: {filename} -> {new_class} by {username}")
                return True
            else:
                print(f"Failed to save validation for {filename}")
                return False
                
        except Exception as e:
            print(f"Error saving to validations table: {str(e)}")
            return False

    def save_validations(self, items, session_username, session_group):
        """Save multiple validation records to validations table. Returns (ok, count)."""
        try:
            count = 0
            for item in items:
                if isinstance(item, dict):
                    query = f"""
                        INSERT INTO {self.validations_table}
                        (file_name, user_group, class_result, place, username_yang_mengubah, timestamp, valid_status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    # Parse timestamp if it's a string
                    timestamp = item.get('timestamp', now_local())
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                        except:
                            timestamp = now_local()
                    
                    params = (
                        item.get('file_name'),
                        item.get('group', session_group),
                        item.get('class_result'),
                        item.get('place'),
                        item.get('username_yang_mengubah', session_username),
                        timestamp,
                        item.get('valid_status', False)
                    )
                    
                    self.db.execute_query(query, params)
                    count += 1
            
            return True, count
            
        except Exception as e:
            print(f"Error in save_validations: {str(e)}")
            return False, str(e)

    def get_monthly_ticket_data(self, year, month, group_id):
        """Get all ticket data for a specific month and year for a group"""
        try:
            # Create date range for the month
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"
            
            query = f"""
                SELECT id, file_name, group_id, class_result, status, place, username, created_at, updated_at,
                       total_process_time, detection_time, classification_time
                FROM {self.ticket_table}
                WHERE group_id = %s 
                AND created_at >= %s 
                AND created_at < %s
                ORDER BY created_at DESC
            """
            
            params = (group_id, start_date, end_date)
            results = self.db.execute_query(query, params, fetch=True)
            
            if results:
                normalized = []
                for result in results:
                    raw_place = result.get('place')
                    place_val = raw_place
                    # Normalize place: '-' stays '-', numeric strings -> int, 0 stays 0
                    if isinstance(raw_place, str):
                        p = raw_place.strip()
                        if p == '-':
                            place_val = '-'
                        else:
                            try:
                                place_val = int(p)
                            except Exception:
                                place_val = p
                    elif isinstance(raw_place, (int, float)):
                        place_val = int(raw_place)
                    
                    normalized.append({
                        'id': result['id'],
                        'file_name': result['file_name'],
                        'group': result['group_id'],
                        'class_result': result['class_result'],
                        'status': result['status'],
                        'place': place_val,
                        'username': result['username'],
                        'timestamp': result['created_at'].strftime('%Y-%m-%d %H:%M:%S') if result['created_at'] else None,
                        'uploader': result['username'],  # Add alias for compatibility
                        'filename': result['file_name'],  # Add alias for compatibility
                        'total_process_time': float(result['total_process_time']) if result['total_process_time'] else 0.0,
                        'detection_time': float(result['detection_time']) if result['detection_time'] else 0.0,
                        'classification_time': float(result['classification_time']) if result['classification_time'] else 0.0
                    })
                
                return normalized
            
            return []
            
        except Exception as e:
            print(f"Error getting monthly ticket data: {str(e)}")
            return []

    def get_yearly_ticket_data(self, year, group_id):
        """Get all ticket data for a specific year and group"""
        try:
            start_date = f"{year}-01-01"
            end_date = f"{year + 1}-01-01"
            
            query = f"""
                SELECT id, file_name, group_id, class_result, status, place, username, created_at, updated_at,
                       total_process_time, detection_time, classification_time
                FROM {self.ticket_table}
                WHERE group_id = %s 
                AND created_at >= %s 
                AND created_at < %s
                ORDER BY created_at DESC
            """
            
            params = (group_id, start_date, end_date)
            results = self.db.execute_query(query, params, fetch=True)
            
            if results:
                normalized = []
                for result in results:
                    raw_place = result.get('place')
                    place_val = raw_place
                    # Normalize place: '-' stays '-', numeric strings -> int, 0 stays 0
                    if isinstance(raw_place, str):
                        p = raw_place.strip()
                        if p == '-':
                            place_val = '-'
                        else:
                            try:
                                place_val = int(p)
                            except Exception:
                                place_val = p
                    elif isinstance(raw_place, (int, float)):
                        place_val = int(raw_place)
                    
                    normalized.append({
                        'id': result['id'],
                        'filename': result['file_name'],
                        'file_name': result['file_name'],
                        'group': result['group_id'],
                        'class_result': result['class_result'],
                        'status': result['status'],
                        'place': place_val,
                        'uploader': result['username'],
                        'username': result['username'],
                        'timestamp': result['created_at'].strftime('%Y-%m-%d %H:%M:%S') if result['created_at'] else None,
                        'created_at': result['created_at'].strftime('%Y-%m-%d %H:%M:%S') if result['created_at'] else None,
                        'updated_at': result['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if result['updated_at'] else None,
                        'total_process_time': float(result['total_process_time']) if result['total_process_time'] else 0.0,
                        'detection_time': float(result['detection_time']) if result['detection_time'] else 0.0,
                        'classification_time': float(result['classification_time']) if result['classification_time'] else 0.0
                    })
                
                return normalized
            
            return []
            
        except Exception as e:
            print(f"Error getting yearly ticket data: {str(e)}")
            return []

    def get_monthly_ticket_count(self, year, month, group_id):
        """Get ticket count for a specific month and year for a group"""
        try:
            # Create date range for the month
            start_date = f"{year}-{month:02d}-01"
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"
            
            query = f"""
                SELECT COUNT(DISTINCT substring(file_name from 1 for 16)) as count
                FROM {self.ticket_table}
                WHERE group_id = %s 
                AND created_at >= %s 
                AND created_at < %s
            """
            
            params = (group_id, start_date, end_date)
            results = self.db.execute_query(query, params, fetch=True)
            
            if results and len(results) > 0:
                return results[0]['count']
            
            return 0
            
        except Exception as e:
            print(f"Error getting monthly ticket count: {str(e)}")
            return 0

    def load_validations(self):
        """Load all validation records from database"""
        try:
            query = f"""
                SELECT id, file_name, user_group, class_result, place, 
                       username_yang_mengubah, timestamp, valid_status
                FROM {self.validations_table}
                ORDER BY timestamp DESC
            """
            
            results = self.db.execute_query(query, fetch=True)
            
            if results:
                return [
                    {
                        'id': result['id'],
                        'file_name': result['file_name'],
                        'group': result['user_group'],
                        'class_result': result['class_result'],
                        'place': result['place'],
                        'username_yang_mengubah': result['username_yang_mengubah'],
                        'timestamp': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if result['timestamp'] else None,
                        'valid_status': result['valid_status']
                    }
                    for result in results
                ]
            
            return []
            
        except Exception as e:
            print(f"Error loading validations: {str(e)}")
            return []