from flask import Flask, request, jsonify, render_template
from query_data2 import query_rag 

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    user_message = request.json.get("message", "")
    chatbot_response = query_rag(user_message)  
    return jsonify({"response": chatbot_response})

if __name__ == "__main__":
    app.run(debug=True)
