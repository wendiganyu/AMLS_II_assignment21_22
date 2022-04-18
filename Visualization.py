"""
Generate the visualization results by loading the saved models. The visualization figures are used for report.
"""
import cv2
'''
    model_G = load_saved_generator("results/gen_bestPSNR_seed13414773155953270327.pth.tar", upscale_factor=2)
    # model_G = load_saved_generator("results/gen_bestPSNR_seed10095113155149744729.pth.tar", upscale_factor=2) #SRResnet

    upscale_factor = 2
    SR_width = 180
    SR_height = 180
    device = torch.device("cpu")
    Tensor = torch.Tensor

    # Get the image input
    # LR_img_path = "Datasets/test/LR_bicubic_X2/0886x2.png"
    LR_img_path = "songtesting.png"
    input_LR_img = img = cv2.imread(LR_img_path).astype(np.float32) / 255
    # input_LR_img_crop = Utils.img_custom_crop(input_LR_img, 990, 990, 73, 83)
    input_LR_img_crop = input_LR_img

    h_SR= 3960
    w_SR = 3960
    reconstruct_SR_image = np.zeros((h_SR, w_SR, 3)).astype(np.float32)
    h, w, c = input_LR_img_crop.shape
    print('width: ', w)
    print('height: ', h)
    print('channel: ', c)

    for i in range(22):
        for j in range(22):
            index_LR_top =  i * 90
            index_LR_left = j * 90
            index_SR_top = i * 180
            index_SR_left = j * 180

            tmp_LR_crop = Utils.img_custom_crop(input_LR_img_crop, SR_height // upscale_factor, SR_width // upscale_factor,
                                                  index_LR_top,
                                                  index_LR_left)

            tmp_LR_crop = cv2.cvtColor(tmp_LR_crop, cv2.COLOR_BGR2RGB)
            tmp_LR_crop = tmp_LR_crop.transpose(2, 0, 1)

            tmp_LR_crop = torch.Tensor(tmp_LR_crop)
            tmp_LR_crop = tmp_LR_crop.unsqueeze(0)

            tmp_SR_area = model_G(tmp_LR_crop)

            tmp_SR_area = tmp_SR_area.detach().numpy()
            tmp_SR_area = tmp_SR_area.astype(np.float32) * 255
            tmp_SR_area = tmp_SR_area.squeeze(0)
            tmp_SR_area = tmp_SR_area.transpose(1, 2, 0)

            reconstruct_SR_image[index_SR_top : index_SR_top + 180, index_SR_left : index_SR_left + 180, :] = tmp_SR_area
            # print(SR_img.shape)
            # cv2.imwrite("testing2.png", cv2.cvtColor(SR_img, cv2.COLOR_RGB2BGR))
    h, w, c = reconstruct_SR_image.shape
    print('width: ', w)
    print('height: ', h)
    print('channel: ', c)
    print(reconstruct_SR_image)
    cv2.imwrite("songtestingtesting.png", cv2.cvtColor(reconstruct_SR_image, cv2.COLOR_RGB2BGR))







    # top_coord: 400, left_coord:510
    input_LR_crop = Utils.img_custom_crop(input_LR_img, SR_height // upscale_factor, SR_width // upscale_factor, 150,
                                          300)

    input_LR_crop = cv2.cvtColor(input_LR_crop, cv2.COLOR_BGR2RGB)
    input_LR_crop = input_LR_crop.transpose(2,0,1)

    input_LR_crop = torch.Tensor(input_LR_crop)
    input_LR_crop = input_LR_crop.unsqueeze(0)

    SR_img = model_G(input_LR_crop)

    SR_img = SR_img.detach().numpy()
    SR_img = SR_img.astype(np.float32) * 255
    SR_img = SR_img.squeeze(0)
    SR_img = SR_img.transpose(1,2,0)
    print(SR_img.shape)
    cv2.imwrite("testing2.png", cv2.cvtColor(SR_img, cv2.COLOR_RGB2BGR))
    '''

HR_img_path = "Datasets/test/HR/0886.png"
HR_img = cv2.imread(HR_img_path)
line_thickness = 4
cv2.line(HR_img, (600, 300), (780, 300), (0,0,255), thickness=line_thickness)
cv2.line(HR_img, (600, 300), (600, 480), (0,0,255), thickness=line_thickness)
cv2.line(HR_img, (780, 300), (780, 480), (0,0,255), thickness=line_thickness)
cv2.line(HR_img, (600, 480), (780, 480), (0,0,255), thickness=line_thickness)

cv2.imwrite("lined_HR_img.png", HR_img)
