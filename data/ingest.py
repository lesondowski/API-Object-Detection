import os
import requests

def download_images(df, save_path):
    for idx, url in enumerate(df):
        try: 
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                file_name = url.split('/')[-1]
                with open(os.path.join(save_path, file_name), 'wb') as f:
                    f.write(response.content)
                    print(f"Downloaded {file_name} ({idx + 1}/{len(df)})")
            else:
                print(f"Failed to download {url}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")


def read_urls_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        urls = [line.strip() for line in file if line.strip()]
    return urls
