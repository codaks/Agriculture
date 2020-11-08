from django.shortcuts import render,redirect


# Create your views here.

def index(request):
    return render(request,"index.html")
def tomato(request):
    if request.method=="POST":
        try:
            print("##################    HERE is THE LIST of OBJECTS       ##################")
            print(request.FILES)
            data = request.FILES['fileUpload']

            #Converted to opencv object
            image =  _grab_image(stream=request.FILES["fileUpload"])
        except Exception as e:
            print("There is an error: ******\n",e)

            
        request.session['image_label'] = "Piyush"
        print("Files on REUEST: ",request.FILES)
        print("#################### Data on POST #################### ",request.session['image_label'])
        return redirect("tomato-area")
    else:
        if request.session.has_key('image_label'):
            print("####################   There exists a key   ####################")
            pass
            # del request.session["patients_data"]
        else:
            request.session['image_label'] = ""
            # del request.session["patients_data"]

        print("####################   Data on GET   ####################  ",request.session['image_label'])
        return render(request,"TomatoDetection.html",{
            'data':request.session['image_label']
        })
'''def maskImage(image):
     
    config = InferenceConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=config)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR,'templates/patients/mask_rcnn_length_0018.h5')
    model.load_weights(path, by_name=True)
    image1 = cv2.resize(image,(512,512))
    
        # Detect objects
    r = model.detect([image1], verbose=0)[0]
        # Color splash




    return display_instances(image, r["rois"], r["masks"], r["class_ids"], class_names=['bg','bacterial spot','Tomato leaf curl',' Septoria leaf spot','Tomato mosaic'],
                            scores=None, title="",
                            figsize=(5, 5), ax=None,
                            show_mask=True, show_bbox=False,
                            captions=None)
                            
                            '''

def _grab_image(path=None, stream=None, url=None):
    
	if path is not None:
		image = cv2.imread(path)
	else:	
        
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()

		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()


		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image


def returnRedirect(reuquest):
    return redirect("home")


def live(request):
    return render(request,"soil-data.html")

##################################################################################################################
##########################################        MASKING CODE        ############################################ 
##################################################################################################################   
'''class lengthConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "length"

    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1
    VALIDATION_STEPS=120
    
    GPU_COUNT =1
    BATCH_SIZE=2
    IMAGE_MAX_DIM=512
    
    STEPS_PER_EPOCH = 300

    # Skip detections with < 90% confidence

class InferenceConfig(lengthConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1



def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):

    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()



def random_colors(N, bright=True):
    
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
   # random.shuffle(colors)
    return colors



def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
   
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
   #ax.imshow(masked_image.astype(np.uint8))
    ################################     update         ##############################    
    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR,'media/scanned_img/')
    path+=file_name
    skimage.io.imsave(path,masked_image.astype(np.uint8))
    return file_name

'''