import cv2

def calculate_average_pixel_value(image):
    height, width, _ = image.shape

    total_r, total_g, total_b = 0, 0, 0
    total_pixels = width * height

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]
            total_r += r
            total_g += g
            total_b += b

    avg_r = total_r // total_pixels
    avg_g = total_g // total_pixels
    avg_b = total_b // total_pixels

    return (avg_r + avg_g + avg_b) / 3

def captchaSplitNoSave(imageBlob): #maybe add saving it here for further testing
    print('a')
    row_length = [[0,83],[86,169],[172,255]]
    col_length = [[0,62],[65,127]]
    imageArray = []
    for row in range(2):
        print('b')
        up = col_length[row][0]
        down = col_length[row][1]
        for col in range(3):
            print('c')
            cropped_image = imageBlob[up:down, row_length[col][0]:row_length[col][1]]
            print('c2')
            cropped_image[4:17, 0:15] = 0
            print('d')
            if calculate_average_pixel_value(cropped_image) > 40:
                print('e')
                topPart = cropped_image[0:4, 0:15]
                bottomPart = cropped_image[17:26, 0:15]
                cropped_image[4:8, 0:15] = topPart
                cropped_image[8:17, 0:15] = bottomPart
                print('f')
            imageArray.append(cropped_image)
    return imageArray


# if __name__ == "__main__": #84x63
#
#     split_image("captcha/1.jpg", "captchaSplitted")
