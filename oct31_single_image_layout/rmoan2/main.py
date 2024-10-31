import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor
from skimage.measure import LineModelND, ransac
import matplotlib.pyplot as plt

from ultralytics import YOLO


def segment_elephant(): 
    model = YOLO('yolov8s-seg.pt')  

    image = cv2.imread("image.png")

    results = model(image)

    for result in results:
        masks = result.masks.data.numpy()  
        classes = result.boxes.cls.numpy()  

        for i, class_id in enumerate(classes):
            if class_id == 20:  # Elephant class id is 20
                mask = masks[i]
                
                # Convert mask to binary image
                binary_mask = (mask < 0.5).astype(np.uint8) * 255

                # Ensure the mask has the same dimensions as the image
                binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
                
                # Apply the mask to the original image
                masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
                
                # Visualize result
                # cv2.imshow("Elephant Segmentation", masked_image)
                # cv2.waitKey(0)

                cv2.imwrite("elephant_mask.png", masked_image)

                return masked_image
    else:
        print("No elephant detected in the image.")

def add_obj(background, img, mask, x, y):
    '''
    source: https://medium.com/@alexppppp/adding-objects-to-image-in-python-133f165b9a01

    Arguments:
    background - background image in CV2 RGB format
    img - image of object in CV2 RGB format
    mask - mask of object in CV2 RGB format
    x, y - coordinates of the center of the object image
    0 < x < width of background
    0 < y < height of background
    
    Function returns background with added object in CV2 RGB format
    
    CV2 RGB format is a numpy array with dimensions width x height x 3
    '''
    
    bg = background.copy()
    
    h_bg, w_bg = bg.shape[0], bg.shape[1]
    
    h, w = img.shape[0], img.shape[1]
    
    # Calculating coordinates of the top left corner of the object image
    x = x - int(h/2)
    y = y - int(w/2)    
    
    mask_boolean = mask[:,:,0] == 0
    mask_rgb_boolean = np.stack([mask_boolean, mask_boolean, mask_boolean], axis=2)
    
    if x >= 0 and y >= 0:
    
        h_part = h - max(0, y+h-h_bg) # h_part - part of the image which overlaps background along y-axis
        w_part = w - max(0, x+w-w_bg) # w_part - part of the image which overlaps background along x-axis

        bg[y:y+h_part, x:x+w_part, :] = bg[y:y+h_part, x:x+w_part, :] * ~mask_rgb_boolean[0:h_part, 0:w_part, :] + (img * mask_rgb_boolean)[0:h_part, 0:w_part, :]
        
    elif x < 0 and y < 0:
        
        h_part = h + y
        w_part = w + x
        
        bg[0:0+h_part, 0:0+w_part, :] = bg[0:0+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[h-h_part:h, w-w_part:w, :] + (img * mask_rgb_boolean)[h-h_part:h, w-w_part:w, :]
        
    elif x < 0 and y >= 0:
        
        h_part = h - max(0, y+h-h_bg)
        w_part = w + x
        
        bg[y:y+h_part, 0:0+w_part, :] = bg[y:y+h_part, 0:0+w_part, :] * ~mask_rgb_boolean[0:h_part, w-w_part:w, :] + (img * mask_rgb_boolean)[0:h_part, w-w_part:w, :]
        
    elif x >= 0 and y < 0:
        
        h_part = h + y
        w_part = w - max(0, x+w-w_bg)
        
        bg[0:0+h_part, x:x+w_part, :] = bg[0:0+h_part, x:x+w_part, :] * ~mask_rgb_boolean[h-h_part:h, 0:w_part, :] + (img * mask_rgb_boolean)[h-h_part:h, 0:w_part, :]
    
    return bg


def load_depth_map_and_normals(file_name):
    # Load depth map and surface normals
    depth_map = np.load('backgrounds/depths/'+file_name+'_pred.npy')
    depth_map_8bit = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    colored_depth_map = cv2.applyColorMap(depth_map_8bit, cv2.COLORMAP_JET)
    surface_normals = np.load('backgrounds/normals/'+file_name+'_pred.npy')

    # Normalize depth map and noramls
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    return colored_depth_map, (surface_normals + 1) / 2

def draw_mask(inserted_img, mask):
    # Draw the mask on the image
    mask = np.expand_dims(mask, axis=2)
    mask = np.repeat(mask, 3, axis=2)
    mask = mask.astype(np.uint8) * 255

    inserted_img = cv2.addWeighted(inserted_img, 1, mask, 0.5, 0)

    return inserted_img

def resize_object(object_img, scale):
    new_height = int(object_img.shape[0] * scale)
    new_width = int(object_img.shape[1] * scale)
    return cv2.resize(object_img, (new_width, new_height))

def get_upward_facing_points(depth_map, surface_normals):
    depth_map = np.load('backgrounds/depths/quad_pred.npy')
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    points_3d = np.dstack((x, y, depth_map)).reshape(-1, 3)

    
    upward_mask = surface_normals[:, :, 1] > 0.7 
    filtered_points = points_3d[upward_mask.flatten()]
    return filtered_points

def fit_ground_plane(filtered_points):
    pos = filtered_points[:, :2]  
    Z = filtered_points[:, 2]   

    # Create the design matrix A
    A = np.vstack([pos[:,0], pos[:,1], np.ones(len(Z))]).T

    # Solve for the coefficients
    beta = np.linalg.lstsq(A, Z, rcond=None)[0]

    a, b, c = beta
    print(f"Ground plane equation: z = {a}x + {b}y + {c}")


    plane_eq = np.array([a, b, -1, c])
    return plane_eq


def estimate_homography(plane_eq, width, height):
    # Assume the principal point is at the image center
    cx, cy = width / 2, height / 2

    a, b, c, d = plane_eq
    H = np.array([[1, 0, -cx],
                  [0, 1, -cy],
                  [0, 0, 1]]).astype(float)

    H[0, 0] = a
    H[1, 1] = b
    H[2, 0] = c
    H[2, 1] = d

    H = H / H[2, 2]  # Normalize the homography matrix
    return H

def reprojection_error(params, H, image_points, world_points):
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float32)  
        HK = (H @ K).astype(np.float32)

        world_points = world_points.astype(np.float32)
        
        projected_points = cv2.perspectiveTransform(world_points.reshape(-1, 1, 2), HK)
        error = np.sum((image_points - projected_points.reshape(-1, 2))**2)
        return error

def estimate_camera_intrinsics(H, width, height):
    # Normalize H
    H_normalized = H / np.linalg.norm(H[:, 0])

    h1, h2, h3 = H_normalized[:, 0], H_normalized[:, 1], H_normalized[:, 2]

    # Initial estimate
    initial_focal_length = min(width, height)*35

    # Initial intrinsic matrix
    K = np.array([[initial_focal_length, 0, width / 2],
                  [0, initial_focal_length, height / 2],
                  [0, 0, 1]])
    
    K = optimize(width, height, H, width / 2, height / 2, initial_focal_length)

    # Decompose homography
    r1 = np.linalg.inv(K) @ h1
    r2 = np.linalg.inv(K) @ h2
    r3 = np.cross(r1, r2)
    T = np.linalg.inv(K) @ h3

    # rotation
    R = np.column_stack((r1, r2, r3))


    # cam height is in translation vector
    camera_height = np.abs(T[2])

    return K, R, camera_height

def optimize(width, height, H, cx, cy, initial_focal_length):
    from scipy.optimize import minimize

    image_points = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    world_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    initial_params = [initial_focal_length, initial_focal_length, cx, cy]
    result = minimize(reprojection_error, initial_params, args=(H, image_points, world_points))

    opt_fx, opt_fy, opt_cx, opt_cy = result.x

    opt_K = np.array([[opt_fx, 0, opt_cx],
                            [0, opt_fy, opt_cy],
                            [0, 0, 1]])
    
    return opt_K


def fit_horizon_using_ground_points(depth_map, normals):

    depth_map = np.load('backgrounds/depths/quad_pred.npy')

    # Identify ground plane using normals
    ground_mask = (np.abs(normals[:, :, 1]) > 0.99)
    
    # Extract depth values for ground pixels
    ground_points = np.argwhere(ground_mask)

    # Get the topmost (smallest y value) ground points whose normals are facing upwards
    upward_facing_ground_points = ground_points[ground_points[:, 0].argsort()][:1000]

    # get the depths of the topmost ground points
    upward_facing_ground_depths = depth_map[upward_facing_ground_points[:, 0], upward_facing_ground_points[:, 1]]


    ground_3d_points = np.hstack((upward_facing_ground_points, upward_facing_ground_depths.reshape(-1, 1)))

    


    model, inliers = ransac(ground_3d_points, LineModelND, min_samples=2,
                            residual_threshold=1, max_trials=1000)

 
    line_y = np.array([0, depth_map.shape[0] - 1])
    line_x = model.predict_y(line_y)

    
    horizon_position = (int(np.mean(line_x)), int(np.mean(line_y)))

    return line_x, line_y, horizon_position

def draw_possible_object_locs(filename):
    depth_map, surface_normals = load_depth_map_and_normals(filename)

    person_height = .4  # Height of the person in world units

    # get the intrinsic matrix
    width, height = depth_map.shape[1], depth_map.shape[0]
    filtered_points = get_upward_facing_points(depth_map, surface_normals)
    plane_eq = fit_ground_plane(filtered_points)
    H = estimate_homography(plane_eq, width, height)
    K, R, camera_height = estimate_camera_intrinsics(H, width, height)


    x,y,horizon_line = fit_horizon_using_ground_points(depth_map, surface_normals)
    horizon_line= horizon_line[1]

    background = cv2.imread('images/quad.jpg')
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

     
    # Draw possible object locations
    count = 0
    while count < 30:
        row,col,val = depth_map.shape
        x = np.random.randint(0,row)
        y = np.random.randint(0,col)


        if depth_map[x,y][0] < 100 or depth_map[x,y][1] > 50 or depth_map[x,y][2] > 50 or surface_normals[x,y][1] < 0.8:
            continue


        person_height_px = int(abs(person_height * (horizon_line - x) / camera_height))
        cv2.line(depth_map, (y, x), (y, x - person_height_px), (0, 0, 0), 10)
        cv2.line(surface_normals, (y, x), (y, x - person_height_px), (0, 0, 0), 10)
        cv2.line(background, (y, x), (y, x - person_height_px), (0, 0, 0), 10)

        count += 1


    import matplotlib.pyplot as plt

    # Visualize depth map
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Depth Map')
    plt.imshow(depth_map)
    plt.colorbar()

    # Visualize surface normals
    plt.subplot(1, 2, 2)
    plt.title('Surface Normals')
    plt.imshow(surface_normals) 
    plt.colorbar()

    plt.show()

    plt.imshow(background)
    plt.show()


def insert_object_into_img(filename):
    depth_map, surface_normals = load_depth_map_and_normals(filename)


    obj_height = 1.5  # Height of the person in world units

    # get the intrinsic matrix
    width, height = depth_map.shape[1], depth_map.shape[0]
    filtered_points = get_upward_facing_points(depth_map, surface_normals)
    plane_eq = fit_ground_plane(filtered_points)
    H = estimate_homography(plane_eq, width, height)
    K, R, camera_height = estimate_camera_intrinsics(H, width, height)

    x,y,horizon_line = fit_horizon_using_ground_points(depth_map, surface_normals)
    horizon_line= horizon_line[1]



    import matplotlib.pyplot as plt
    # Display the depth map with the estimated horizon line
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map)
    plt.plot(x, y, 'black', label="Horizon Line")  
    plt.title("Estimated Horizon Line on Depth Map")
    plt.legend()
    # make the range of the x and y axis match the image size
    plt.xlim(0, depth_map.shape[1])
    plt.ylim(depth_map.shape[0], 0)
    plt.show()

     
    # Draw possible object locations
    done = False
    while not done:
        row,col,val = depth_map.shape
        x = np.random.randint(0,row)
        y = np.random.randint(0,col)


        if depth_map[x,y][0] < 100 or depth_map[x,y][1] > 50 or depth_map[x,y][2] > 50 or surface_normals[x,y][1] < 0.8:
            continue

    
        obj_height_px = int(abs(obj_height * (horizon_line - x) / camera_height))
        

        cv2.line(depth_map, (y, x), (y, x - obj_height_px), (0, 0, 0), 10)
        cv2.line(surface_normals, (y, x), (y, x - obj_height_px), (0, 0, 0), 10)

        # Original image, which is the background 
        background = cv2.imread('images/quad.jpg')
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        # Image of the object
        img = cv2.imread('image.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Image the object's mask
        mask = segment_elephant()

        # scale the image and the mask so that the mask is obj_height_px pixels tall
        scale = obj_height_px / mask.shape[0]
        img = resize_object(img, scale)
        mask = resize_object(mask, scale)

        composition_1 = add_obj(background, img, mask, y, x)
        plt.figure(figsize=(15,15))
        plt.imshow(composition_1)

        done = True



    

if __name__ == "__main__":
    depth_map, surface_normals = load_depth_map_and_normals('quad')

    draw_possible_object_locs('quad')

    # insert_object_into_img('quad')


    

