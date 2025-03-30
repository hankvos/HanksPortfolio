from flask import Flask
import os  # Added this!

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello, Property Analyzer is live!"

if os.environ.get("RENDER"):
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
else:
    app.run(debug=True)