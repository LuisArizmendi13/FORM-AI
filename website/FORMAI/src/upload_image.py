import requests
import os

def upload_file_to_server(file_path, server_url="http://127.0.0.1:5000/upload"):
    """
    Uploads a file to the Flask server and returns its hosted URL.
    """
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return None

    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(server_url, files=files)

        if response.status_code == 200:
            return response.json().get('url')
        else:
            print(f"❌ Error uploading file: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"⚠️ Exception during upload: {e}")
        return None

# Example usage
if __name__ == "__main__":
    url = upload_file_to_server("earth.gif")
    if url:
        print("✅ Uploaded to:", url)
    else:
        print("❌ Upload failed.")
