"""
SQLite Database Setup for LangGraph RAG System
Creates the incidents database with sample data for testing autonomous agentic RAG.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime, timedelta
import random

def create_incidents_database(db_path: str):
    """Create incidents database with sample data."""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create incidents table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            severity TEXT NOT NULL CHECK(severity IN ('Low', 'Medium', 'High', 'Critical')),
            status TEXT NOT NULL CHECK(status IN ('Open', 'In Progress', 'Resolved', 'Closed')),
            assigned_to TEXT,
            category TEXT NOT NULL,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolution_notes TEXT
        )
    ''')
    
    # Sample incident data for testing autonomous agentic RAG
    sample_incidents = [
        {
            'title': 'Network Connectivity Issues in Building A',
            'description': 'Multiple users in Building A are experiencing intermittent network connectivity issues. The problem appears to be related to the main router configuration. Users report slow loading times and frequent disconnections from the corporate VPN.',
            'severity': 'High',
            'status': 'In Progress',
            'assigned_to': 'Network Team',
            'category': 'Infrastructure',
            'resolution_notes': 'Investigating router firmware update requirements.'
        },
        {
            'title': 'Email Server Performance Degradation',
            'description': 'The email server is experiencing significant performance degradation during peak hours (9 AM - 11 AM). Users report delayed email delivery and timeout errors when accessing mailboxes. The issue may be related to insufficient server resources.',
            'severity': 'Medium',
            'status': 'Open',
            'assigned_to': 'Systems Administrator',
            'category': 'Email Systems',
            'resolution_notes': None
        },
        {
            'title': 'Security Breach Attempt Detected',
            'description': 'Automated security systems detected multiple failed login attempts from suspicious IP addresses targeting admin accounts. The attacks appear to be coordinated and are using dictionary-based password attacks. Immediate action required to secure affected accounts.',
            'severity': 'Critical',
            'status': 'Resolved',
            'assigned_to': 'Security Team',
            'category': 'Security',
            'resolution_notes': 'All affected accounts secured. IP addresses blocked. Security protocols updated.'
        },
        {
            'title': 'Database Query Performance Issues',
            'description': 'Production database queries are running significantly slower than normal. Analysis shows that several large tables lack proper indexing, and there may be a need for database optimization. This is affecting multiple applications that depend on the database.',
            'severity': 'High',
            'status': 'In Progress',
            'assigned_to': 'Database Administrator',
            'category': 'Database',
            'resolution_notes': 'Index analysis in progress. Optimization plan being developed.'
        },
        {
            'title': 'Printer Driver Installation Failure',
            'description': 'Several workstations cannot install the new printer drivers for the HP LaserJet Pro series. The installation fails with error code 0x80070003. This affects approximately 25 users in the accounting department who need to print financial reports.',
            'severity': 'Low',
            'status': 'Closed',
            'assigned_to': 'IT Support',
            'category': 'Hardware',
            'resolution_notes': 'Driver compatibility issue resolved. Updated driver package deployed successfully.'
        },
        {
            'title': 'Backup System Failure',
            'description': 'The automated backup system failed to complete the nightly backup process for the past three days. Error logs indicate issues with the backup storage array. Critical data backup is at risk, requiring immediate attention to prevent data loss.',
            'severity': 'Critical',
            'status': 'Open',
            'assigned_to': 'Backup Team',
            'category': 'Backup Systems',
            'resolution_notes': None
        },
        {
            'title': 'Software License Compliance Issue',
            'description': 'Internal audit revealed that several departments are using software without proper licensing. This includes Adobe Creative Suite and Microsoft Office Professional licenses. Legal compliance team needs to address this immediately to avoid penalties.',
            'severity': 'Medium',
            'status': 'In Progress',
            'assigned_to': 'Compliance Team',
            'category': 'Software Licensing',
            'resolution_notes': 'License audit in progress. Purchase orders being prepared for required licenses.'
        },
        {
            'title': 'Wi-Fi Access Point Malfunction',
            'description': 'The Wi-Fi access point in Conference Room B is malfunctioning, providing weak signal strength and frequent disconnections. This is impacting important client meetings and presentations. The issue may be hardware-related.',
            'severity': 'Medium',
            'status': 'Resolved',
            'assigned_to': 'Network Technician',
            'category': 'Network Infrastructure',
            'resolution_notes': 'Defective access point replaced. Signal strength now optimal.'
        },
        {
            'title': 'Application Server Memory Leak',
            'description': 'The main application server is experiencing a memory leak that causes gradual performance degradation and eventually leads to application crashes. The issue appears to be related to a recent software update. Server restarts are required every 6-8 hours.',
            'severity': 'High',
            'status': 'Open',
            'assigned_to': 'Application Team',
            'category': 'Application Server',
            'resolution_notes': None
        },
        {
            'title': 'VPN Connection Timeout Issues',
            'description': 'Remote workers are experiencing frequent VPN connection timeouts, especially during video conferences. The issue seems to be more prevalent during peak usage hours. This affects productivity for remote and hybrid workers.',
            'severity': 'Medium',
            'status': 'In Progress',
            'assigned_to': 'Network Security Team',
            'category': 'VPN Systems',
            'resolution_notes': 'Load balancing configuration being optimized.'
        }
    ]
    
    # Insert sample data
    for incident in sample_incidents:
        cursor.execute('''
            INSERT INTO incidents (title, description, severity, status, assigned_to, category, resolution_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            incident['title'],
            incident['description'], 
            incident['severity'],
            incident['status'],
            incident['assigned_to'],
            incident['category'],
            incident['resolution_notes']
        ))
    
    conn.commit()
    conn.close()
    
    print(f"✅ SQLite database created successfully at: {db_path}")
    print(f"✅ Inserted {len(sample_incidents)} sample incidents")
    return db_path

if __name__ == "__main__":
    # Default path
    db_path = os.path.join("data", "incidents.db")
    create_incidents_database(db_path)
