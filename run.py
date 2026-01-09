from backend.app import app

if __name__ == '__main__':
    print("="*60)
    print("SERVER STARTING")
    print("Open: http://localhost:5000")
    print("="*60)
    app.run(debug=True, port=5000)
