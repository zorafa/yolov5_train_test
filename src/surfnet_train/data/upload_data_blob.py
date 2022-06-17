import os
from azure.storage.blob import BlobServiceClient
import argparse
from tqdm import tqdm


def upload_file(img_path, blob_service_client, container):
    label_path = img_path.split('/')
    label_path = label_path[:-2] + ['labels'] + [label_path[-1].split('.')[0] + '.txt']
    label_path = '/'.join(label_path)
    
    blob_client_img = blob_service_client.get_blob_client(container=f'{container}',
                                                      blob='data/images/'+img_path.split('/')[-1])
    blob_client_label = blob_service_client.get_blob_client(container=f'{container}',
                                                      blob='data/labels/'+label_path.split('/')[-1])
    
    with open(img_path, 'rb') as img:
        blob_client_img.upload_blob(img, overwrite=True)
        
    with open(label_path, 'rb') as label:
        blob_client_label.upload_blob(label, overwrite=True)
        
def upload_files(img_directory, blob_service_client, container):
    img_paths = os.listdir(img_directory)
    
    
    for img_path in tqdm(img_paths):
        img_path = os.path.join(img_directory,img_path)
        upload_file(img_path, blob_service_client, container)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_directory',type=str)
    parser.add_argument('--connection_string',type=str)
    parser.add_argument('--container', type=str)
    args = parser.parse_args() 
    blob_service_client = BlobServiceClient.from_connection_string(args.connection_string)
    upload_files(img_directory=args.img_directory,
                 blob_service_client=blob_service_client,
                 container=args.container)
    

