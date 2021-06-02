LATENT_DIM = 432
CHANNELS = 3

def create_generator():
    gen_input = Input(shape=(LATENT_DIM, ))

    x = Dense(512 * 16 * 16)(gen_input)#128
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 512))(x)

    x = Conv2D(512, 5, padding='same')(x)#256
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator

#####
def detect_face(image):
    cascPath = "haarcascade_frontalface_default.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

    # Read the image
        
    image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)#((image + 1) * 225 / 2).round()
    image = image.astype(np.uint8)

    gray = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    #    gray,
    #    scaleFactor=1.1,
    #    minNeighbors=5,
    #    minSize=(30, 30),
    #    flags = cv2.CASCADE_SCALE_IMAGE #cv2.cv.CV_HAAR_SCALE_IMAGE
    #)

    #print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    face = []
    if len(faces)==0:
        orig_faces.append([face,(1,1,1,1)])
    for (x, y, w, h) in faces:
        face = image[x:x+w, y:y+h]

        if face is None:
            orig_faces.append([[], (x, y, x+w, y+h)])
        else:
            orig_faces.append([face, (x, y, x+w, y+h)])
        break    
#####

#####
import time
iters = 15000
batch_size = 1

RES_DIR = 'res2'
FILE_PATH = '%s/generated_%d.png'
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

CONTROL_SIZE_SQRT = 6
control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT**2, LATENT_DIM)) / 2

start = 0
d_losses = []
a_losses = []
images_saved = 0

#gan.load_weights("/gan.h5")

#images = images[:2]

for step in range(iters):
    start_time = time.time()
    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
     
    real = images.copy()[:2]#[np.random.choice(len(images), size=batch_size, replace=False)]#images[start:start + batch_size]
    orig = real.copy()
    orig_faces = []
    for img in orig:
        detect_face(img)

    faces = [face for face, coord in orig_faces if len(face) != 0]
    coords = [coord for face, coord in orig_faces if len(face) != 0]
    
    random_faces = orig_faces.copy()
    random_faces = [random_faces[1], random_faces[0]]
    #np.random.shuffle(random_faces)
    
    rand_faces = [face for face, coord in random_faces if len(face) != 0]
    rand_coords = [coord for face, coord in random_faces if len(face) != 0]

    ######
    for face in range(len(faces)):
        src = faces[face]
        new = rand_faces[face]       
        
        x1,y1,x2,y2 = coords[face]
        #calculate the 50 percent of original dimensions
        width = int(orig[face][x1:x2, y1:y2].shape[1])
        height = int(orig[face][x1:x2, y1:y2].shape[0])

        # dsize
        dsize = (width, height)

        # resize image
        output = cv2.resize(new, dsize)

        #output = np.array(output, dtype="uint8")
        
        rand_faces[face] = output
        #x1,y1,x2,y2 = coords[face]
        #faces[face] = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        
        output = cv2.normalize(output, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        output = output.astype(np.float32)
        orig[face][x1:x2, y1:y2] = output
    
        faces[face] = cv2.resize(faces[face], (12, 12)).flatten().reshape(1,-1)

    ###### 
    generated = generator.predict(faces[0])

    real = [np.random.choice(len(images), size=batch_size, replace=False)]#images[start:start + batch_size]
    combined_images = np.concatenate([generated, real])

    labels = np.concatenate([np.ones((1, 1)), np.zeros((batch_size, 1))])
    labels += .05 * np.random.random(labels.shape)
    
    d_loss = discriminator.train_on_batch(combined_images, labels)
    d_losses.append(d_loss)

    latent_vectors = np.random.normal(size=(batch_size, LATENT_DIM))
    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
    a_losses.append(a_loss)

    start += batch_size
    if start > images.shape[0] - batch_size:
        start = 0

    if step % 8 == 0:#49:
        gan.save_weights('/gan.h5')

        plt.figure(1, figsize=(10, 10))
        for i in range(1):
            plt.subplot(5, 5, i+1)
            plt.imshow(generated[i])
            plt.axis('off')
        plt.show()  
    
        print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (step + 1, iters, d_loss, a_loss, time.time() - start_time))

        #control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
        #control_generated = generator.predict(control_vectors)
        
        #for i in range(CONTROL_SIZE_SQRT ** 2):
        #    x_off = i % CONTROL_SIZE_SQRT
        #    y_off = i // CONTROL_SIZE_SQRT
        #    control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(y_off + 1) * HEIGHT, :] = control_generated[i, :, :, :]
        #im = Img.fromarray(np.uint8(control_image * 255))#.save(StringIO(), 'jpeg')
        #im.save(FILE_PATH % (RES_DIR, images_saved))
        #images_saved += 1


#####

