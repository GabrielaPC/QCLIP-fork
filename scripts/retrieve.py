from modules.symbolic import text_processor
from modules.utils.general import get_device
from modules.data_pipeline.embeddings import ImgStream, embed_images
from modules.data_pipeline.retrieval import BaseDatasetRetriever
from pathlib import Path
import os

if __name__ == '__main__':
    DATASET = 'aro'
    PARSER_PATH = '/cs/research/pplv/comp_bridge/bobcat'
    ROOT_PATH = os.getcwd()
    OUT_DIR = ROOT_PATH + '/data/' + DATASET + '/processed'
    IMG_PATH = ROOT_PATH + '/data/' + DATASET + '/raw/images.zip'
    DEV = get_device()

    print(f"Retrieving {DATASET} dataset from {ROOT_PATH}/data...")
    retriever = BaseDatasetRetriever(dataset_name=DATASET, data_root=ROOT_PATH+'/data')
    retriever.retrieve()

    print(f"Processing columns: {', '.join(retriever.text_labels)}...")
    print(f"Backend: {DEV}, Batch size: {64}, Output Directory: {PARSER_PATH}")
    functor = text_processor.TextProcessor(PARSER_PATH, 64, DEV)
    functor.text2diagram(path=OUT_DIR, dataset=retriever.data, text_labels=retriever.text_labels)

    #fname = os.path.basename(f)
    generator = ImgStream(IMG_PATH, file_type='zip')
    embed_images(generator, f'{OUT_DIR}/images_embeddings.pt', device=DEV)
    # for f in os.listdir(IMG_PATH):
    #     if f.endswith('.zip'):
    #         fname = os.path.basename(f)
    #         generator = ImgStream(IMG_PATH, file_type='zip')
    #         embed_images(generator, f'{OUT_DIR}/images/{fname}_embeddings.pt', device=DEV)