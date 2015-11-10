#ifndef _CCV_IMAGE_H_
#define _CCV_IMAGE_H_

// Matrix Structure declaration
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int channels;
    unsigned int pitch;
    float * elements;
} ccvImage;

#define ccvImage_channels 3

#define ccvImage_getWidth(img) ((img)->width)
#define ccvImage_getHeight(img) ((img)->height)
#define ccvImage_getChannels(img) ((img)->channels)
#define ccvImage_getPitch(img) ((img)->pitch)
#define ccvImage_getData(img) ((img)->data)

#define ccvImage_setWidth(img, val) (ccvImage_getWidth(img) = val)
#define ccvImage_setHeight(img, val) (ccvImage_getHeight(img) = val)
#define ccvImage_setChannels(img, val) (ccvImage_getChannels(img) = val)
#define ccvImage_setPitch(img, val) (ccvImage_getPitch(img) = val)
#define ccvImage_setData(img, val) (ccvImage_getData(img) = val)

#endif // _CCV_IMAGE_H_