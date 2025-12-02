import os
from azure.storage.blob import ContainerClient

sas_url = "https://aimistanforddatasets01.blob.core.windows.net/chexlocalize?sv=2019-02-02&sr=c&sig=VH82QjAUf7kibsDLwJ%2FsBPhUS%2Fx8iqufyBAU0HvaNso%3D&st=2025-11-30T21%3A31%3A56Z&se=2025-12-30T21%3A36%3A56Z&sp=rl"

print("Connecting to container...")
container = ContainerClient.from_container_url(sas_url)

print("Listing blobs in container...")
blob_list = list(container.list_blobs())
print(f"Found {len(blob_list)} file(s) in the container.")

for blob in blob_list:
    blob_name = blob.name
    print("Downloading:", blob_name)

    # Create directories if needed
    local_path = os.path.join("chexlocalize_download", blob_name)
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    # Download
    blob_client = container.get_blob_client(blob_name)
    with open(local_path, "wb") as f:
        data = blob_client.download_blob()
        f.write(data.readall())

print("✅ All files downloaded successfully!")
