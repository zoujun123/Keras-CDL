from utils import *
from CDL import CollaborativeDeepLearning

def main():
    train_mat = read_rating('data/ml-10m/normalTrain.csv')
    test_mat = read_rating('data/ml-10m/test.csv')
    item_mat = read_feature('data/ml-10m/itemFeat.csv')
    num_user = int(train_mat[:,0].max()) + 1
    num_item_feat = item_mat.shape[1]

    model = CollaborativeDeepLearning(num_user, [num_item_feat, 10, 8])
    model.pretrain(item_mat, encoder_noise=0.01, nb_epoch=10)
    model.fineture(train_mat, test_mat, item_mat, reg=0.000000001)

if __name__ == "__main__":
    main()