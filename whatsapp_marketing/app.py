from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import csv
import io
from models import get_db_connection, init_db
import requests
import sqlite3
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey') # Replace with a secure key in production

# Initialize database on startup
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    conn = get_db_connection()
    if request.method == 'POST':
        phone_number_id = request.form['phone_number_id']
        access_token = request.form['access_token']
        business_account_id = request.form['business_account_id']

        # Check if settings exist
        existing = conn.execute('SELECT * FROM settings').fetchone()
        if existing:
            conn.execute('UPDATE settings SET phone_number_id = ?, access_token = ?, business_account_id = ? WHERE id = ?',
                         (phone_number_id, access_token, business_account_id, existing['id']))
        else:
            conn.execute('INSERT INTO settings (phone_number_id, access_token, business_account_id) VALUES (?, ?, ?)',
                         (phone_number_id, access_token, business_account_id))
        conn.commit()
        flash('Settings saved successfully!', 'success')
        return redirect(url_for('settings'))

    settings = conn.execute('SELECT * FROM settings').fetchone()
    conn.close()
    return render_template('settings.html', settings=settings)

@app.route('/contacts', methods=['GET', 'POST'])
def contacts():
    conn = get_db_connection()
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            csv_input = csv.reader(stream)

            # Check for header
            first_row = next(csv_input, None)
            if first_row:
                # If first row looks like a header (e.g. contains letters like 'name', 'phone')
                if not any(char.isdigit() for char in first_row[1]):
                    pass # It's likely a header, so we just continue
                else:
                    # It might be a data row, let's process it
                    if len(first_row) >= 2:
                        try:
                            conn.execute('INSERT INTO contacts (name, phone_number) VALUES (?, ?)', (first_row[0].strip(), first_row[1].strip()))
                        except sqlite3.IntegrityError:
                            pass

            count = 0
            for row in csv_input:
                if len(row) >= 2:
                    name, phone = row[0].strip(), row[1].strip()
                    try:
                        conn.execute('INSERT INTO contacts (name, phone_number) VALUES (?, ?)', (name, phone))
                        count += 1
                    except sqlite3.IntegrityError:
                        pass # Ignore duplicates based on unique phone number
            conn.commit()
            flash(f'Successfully imported {count} contacts!', 'success')
            return redirect(url_for('contacts'))
        else:
            flash('Please upload a CSV file.', 'danger')

    contacts = conn.execute('SELECT * FROM contacts').fetchall()
    conn.close()
    return render_template('contacts.html', contacts=contacts)


@app.route('/templates', methods=['GET', 'POST'])
def templates():
    conn = get_db_connection()
    if request.method == 'POST':
        name = request.form['name']
        language = request.form['language']
        # In a real scenario, you'd make an API call to Meta to create the template here.
        # For demonstration, we save it locally as 'pending'
        conn.execute('INSERT INTO templates (name, language, status) VALUES (?, ?, ?)', (name, language, 'pending'))
        conn.commit()
        flash('Template created and pending approval.', 'success')
        return redirect(url_for('templates'))

    templates = conn.execute('SELECT * FROM templates').fetchall()
    conn.close()
    return render_template('templates.html', templates=templates)

@app.route('/campaigns', methods=['GET', 'POST'])
def campaigns():
    conn = get_db_connection()
    if request.method == 'POST':
        template_id = request.form['template_id']
        template = conn.execute('SELECT * FROM templates WHERE id = ?', (template_id,)).fetchone()
        settings = conn.execute('SELECT * FROM settings').fetchone()

        if not settings or not settings['phone_number_id'] or not settings['access_token']:
            flash('Please configure settings first.', 'danger')
            return redirect(url_for('settings'))

        if template:
            contacts = conn.execute('SELECT * FROM contacts').fetchall()
            url = f"https://graph.facebook.com/v19.0/{settings['phone_number_id']}/messages"
            headers = {
                "Authorization": f"Bearer {settings['access_token']}",
                "Content-Type": "application/json"
            }
            success_count = 0
            for contact in contacts:
                data = {
                    "messaging_product": "whatsapp",
                    "to": contact['phone_number'],
                    "type": "template",
                    "template": {
                        "name": template['name'],
                        "language": {
                            "code": template['language']
                        }
                    }
                }
                # Simulate API call for now unless real credentials are used
                try:
                    response = requests.post(url, headers=headers, json=data)
                    if response.status_code == 200:
                        success_count += 1
                except Exception as e:
                    print(f"Error sending message: {e}")

            flash(f'Campaign sent to {success_count} contacts using template {template["name"]}.', 'success')
        return redirect(url_for('campaigns'))

    templates = conn.execute("SELECT * FROM templates WHERE status != 'rejected'").fetchall()
    conn.close()
    return render_template('campaigns.html', templates=templates)

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        mode = request.args.get('hub.mode')
        token = request.args.get('hub.verify_token')
        challenge = request.args.get('hub.challenge')

        VERIFY_TOKEN = os.environ.get('WEBHOOK_VERIFY_TOKEN', 'my_webhook_secret')

        if mode and token:
            if mode == 'subscribe' and token == VERIFY_TOKEN:
                print('WEBHOOK_VERIFIED')
                return challenge, 200
            else:
                return 'Forbidden', 403
        return 'Invalid request', 400

    if request.method == 'POST':
        body = request.get_json()
        print("Received webhook payload:", body)
        return jsonify({'status': 'success'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=3000)
