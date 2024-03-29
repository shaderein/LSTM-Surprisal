import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def load_text_path(paraphrase=False):
    data_dir = './data'
    story_path = os.path.join(data_dir, "text",
                          'Carver_Full Story_So much water so close to home.docx')
    article_path = os.path.join(
        data_dir, "text", 'Can this marriage be saved -  APA Sicence Watch.docx')

    if paraphrase:
        story_pool_path = os.path.join(
            data_dir, "pools",
            'paraphrased_RainyDayStory_pool.xlsx')

        article_pool_path = os.path.join(
            data_dir, 'pools',
            'paraphrased_APAMarriageArticle_pool.xlsx')
    else:
        story_pool_path = os.path.join(
            data_dir, "pools",
            'Semantically-Related Interruptions_Carver.xlsx')

        article_pool_path = os.path.join(
            data_dir, 'pools',
            'diverseSim_interruptions_APAMarriageArticle_pool_brown_allCatges_seed_1_v2.xlsx')

    return story_path, article_path, \
            story_pool_path, article_pool_path

def load_models(sizes):
    models_path = {100:'./data/LSTM_40m/LSTM_100_40m_a_0-d0.2.pt',
                    400:'./data/LSTM_40m/LSTM_400_40m_a_10-d0.2.pt',
                    1600:'./data/LSTM_40m/LSTM_1600_40m_a_20-d0.2.pt'}

    models = []
    for size in sizes:
        path = models_path[size]
        model = torch.load(path, map_location=torch.device(device))
        model.eval()
        models.append(model)

    return models