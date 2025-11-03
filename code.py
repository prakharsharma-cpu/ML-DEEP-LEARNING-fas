from flask import Flask, request, redirect, url_for, Markup

app = Flask(__name__)

# In-memory "database" to store posts
posts = [
    {"id": 1, "title": "Welcome to My Flask Blog", "content": "This is the very first post of the blog!"},
    {"id": 2, "title": "Flask is Fun!", "content": "Flask makes Python web development simple and flexible."}
]

# --- Base Style and Layout Function ---
base_style = """
<style>
body {
    font-family: 'Segoe UI', Tahoma, sans-serif;
    background: #f5f6fa;
    margin: 40px auto;
    width: 80%;
    color: #333;
}
nav {
    margin-bottom: 20px;
    background: #007bff;
    padding: 10px 20px;
    border-radius: 10px;
}
nav a {
    text-decoration: none;
    color: white;
    font-weight: bold;
    margin-right: 15px;
}
nav a:hover {
    text-decoration: underline;
}
.container {
    background: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
h1, h2 {
    color: #222;
}
ul {
    list-style: none;
    padding: 0;
}
ul li {
    background: #f8f9fa;
    margin: 10px 0;
    padding: 10px 15px;
    border-radius: 8px;
}
ul li:hover {
    background: #eef2f7;
}
input[type=text], textarea {
    width: 100%;
    padding: 10px;
    border-radius: 6px;
    border: 1px solid #ccc;
    margin-top: 6px;
}
button {
    padding: 10px 18px;
    border: none;
    background-color: #007bff;
    color: white;
    border-radius: 6px;
    cursor: pointer;
}
button:hover {
    background-color: #0056b3;
}
.alert {
    color: red;
}
footer {
    margin-top: 30px;
    text-align: center;
    font-size: 14px;
    color: #777;
}
</style>
"""

def layout(content):
    """Wraps HTML content in a consistent layout."""
    return f"""
    <html>
    <head>
        <title>Flask Blog</title>
        {base_style}
    </head>
    <body>
        <nav>
            <a href="/">🏠 Home</a>
            <a href="/add">➕ Add Post</a>
        </nav>
        <div class="container">
            {content}
        </div>
        <footer>
            <p>Powered by Flask 🐍 | Simple Blog Example</p>
        </footer>
    </body>
    </html>
    """

# --- Routes ---
@app.route('/')
def home():
    """Homepage listing all posts."""
    if not posts:
        content = "<h2>No posts yet. <a href='/add'>Add one!</a></h2>"
    else:
        post_links = "<ul>" + "".join(
            f"<li><a href='/post/{p['id']}'><strong>{p['title']}</strong></a></li>"
            for p in posts
        ) + "</ul>"
        content = f"<h2>All Posts</h2>{post_links}"
    return layout(content)

@app.route('/add', methods=['GET', 'POST'])
def add_post():
    """Page to add a new post."""
    msg = ""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()

        if not title or not content:
            msg = "<p class='alert'>⚠️ Both title and content are required.</p>"
        else:
            new_id = len(posts) + 1
            posts.append({"id": new_id, "title": title, "content": content})
            return redirect(url_for('home'))

    form_html = f"""
    <h2>Add a New Post</h2>
    {msg}
    <form method="POST">
        <label>Title:</label><br>
        <input type="text" name="title" placeholder="Enter post title" required><br><br>
        <label>Content:</label><br>
        <textarea name="content" rows="6" placeholder="Write your post here..." required></textarea><br><br>
        <button type="submit">Add Post</button>
    </form>
    """
    return layout(form_html)

@app.route('/post/<int:post_id>')
def post_detail(post_id):
    """Page showing post details."""
    post = next((p for p in posts if p["id"] == post_id), None)
    if not post:
        return layout("<h2>❌ Post not found!</h2><a href='/'>Go Back</a>")

    content = f"""
    <h2>{post['title']}</h2>
    <p>{Markup(post['content']).unescape()}</p>
    <a href="/">← Back to Home</a>
    """
    return layout(content)

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)

pip install flask markupsafe

python app.py
