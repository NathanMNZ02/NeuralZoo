import os
import urllib.request

def download_ishape():
    """
    Avvia il download del dataset iShape, un dataset contenente immagini con mappe di segmentazione complesse.
    """
    train_url = "http://113.44.140.251:9000/ishape/ishape_dataset.tar"
    
    out_dir = "dataset_zoo/coco_format"
    urllib.request.urlretrieve(train_url, os.path.join(out_dir, "ishape_dataset.tar"))