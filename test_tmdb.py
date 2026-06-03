import requests

TMDB_KEY = "0236e660a30a82dfc93ee6e8da9a5605"

print("Step 1: Testing TMDB search API...")
try:
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query=Toy+Story"
    r = requests.get(url, timeout=8)
    data = r.json()
    poster_path = data["results"][0]["poster_path"]
    print(f"✅ Search OK — poster_path: {poster_path}")
except Exception as e:
    print(f"❌ Search failed: {e}")
    exit()

print("\nStep 2: Testing TMDB image download...")
try:
    img_url = f"https://image.tmdb.org/t/p/w342{poster_path}"
    print(f"   Fetching: {img_url}")
    r2 = requests.get(img_url, timeout=8)
    print(f"   Status: {r2.status_code}")
    print(f"   Content-Type: {r2.headers.get('Content-Type')}")
    print(f"   Size: {len(r2.content)} bytes")
    if r2.status_code == 200 and len(r2.content) > 1000:
        print("✅ Image download OK!")
    else:
        print("❌ Image download failed or empty!")
except Exception as e:
    print(f"❌ Image download error: {e}")
