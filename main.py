from Train_Model_GAN import train_GAN_model
from Train_Model_SRResnet import train_SRResnet_model

if __name__ == '__main__':
    # Create the dataset paths of different tracks
    train_LR_track_paths = ["Datasets/train/LR_bicubic_X2", "Datasets/train/LR_bicubic_X3",
                            "Datasets/train/LR_bicubic_X4",
                            "Datasets/train/LR_unknown_X2", "Datasets/train/LR_unknown_X3",
                            "Datasets/train/LR_unknown_X4"]

    valid_LR_track_paths = ["Datasets/valid/LR_bicubic_X2", "Datasets/valid/LR_bicubic_X3",
                            "Datasets/valid/LR_bicubic_X4",
                            "Datasets/valid/LR_unknown_X2", "Datasets/valid/LR_unknown_X3",
                            "Datasets/valid/LR_unknown_X4"]

    test_LR_track_paths = ["Datasets/test/LR_bicubic_X2", "Datasets/test/LR_bicubic_X3", "Datasets/test/LR_bicubic_X4",
                           "Datasets/test/LR_unknown_X2", "Datasets/test/LR_unknown_X3", "Datasets/test/LR_unknown_X4"]

    track_name_list = ["BicubicX2", "BicubicX3", "BicubicX4", "UnknownX2", "UnknownX3", "UnknownX4"]

    upscale_factor_list = [2, 3, 4, 2, 3, 4]

    for index in range(len(train_LR_track_paths)):
        train_SRResnet_model(train_LR_track_paths[index], valid_LR_track_paths[index], test_LR_track_paths[index],
                             upscale_factor_list[index], track_name_list[index])

        train_GAN_model(train_LR_track_paths[index], valid_LR_track_paths[index], test_LR_track_paths[index],
                        upscale_factor_list[index], track_name_list[index])
