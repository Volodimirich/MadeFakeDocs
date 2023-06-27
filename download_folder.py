import gdown, sys

url = sys.argv[1]
if url.split('/')[-1] == '?usp=sharing':
  url= url.replace('?usp=sharing','')
	
gdown.download_folder(url, output='/home/d.maximov/data/models/t5_1.7B/checkpoint-0488', quiet=False)