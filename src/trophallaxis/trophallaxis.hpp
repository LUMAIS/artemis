#pragma once
#include <iostream>
#ifdef __cplusplus
extern "C" {
#endif

#define TROPHALAXIS_EXPORT __attribute__((visibility("default")))


#include <stdlib.h>
#include <pthread.h>

/*
typedef struct zarray zarray_t;
struct zarray
{
    size_t el_sz; // size of each element

    int size; // how many elements?
    int alloc; // we've allocated storage for how many elements?
    char *data;
};*/

TROPHALAXIS_EXPORT  void testtorch();


typedef struct libtorchtrophallaxis_detector libtorchtrophallaxis_detector_t;
struct libtorchtrophallaxis_detector
{
    /*
    ///////////////////////////////////////////////////////////////
    // User-configurable parameters.
    */
    // How many threads should be used?
    int nthreads;

    /*

    // detection of quads can be done on a lower-resolution image,
    // improving speed at a cost of pose accuracy and a slight
    // decrease in detection rate. Decoding the binary payload is
    // still done at full resolution. .
    float quad_decimate;

    // What Gaussian blur should be applied to the segmented image
    // (used for quad detection?)  Parameter is the standard deviation
    // in pixels.  Very noisy images benefit from non-zero values
    // (e.g. 0.8).
    float quad_sigma;

    // When non-zero, the edges of the each quad are adjusted to "snap
    // to" strong gradients nearby. This is useful when decimation is
    // employed, as it can increase the quality of the initial quad
    // estimate substantially. Generally recommended to be on (1).
    //
    // Very computationally inexpensive. Option is ignored if
    // quad_decimate = 1.
    int refine_edges;

    // How much sharpening should be done to decoded images? This
    // can help decode small tags but may or may not help in odd
    // lighting conditions or low light conditions.
    //
    // The default value is 0.25.
    double decode_sharpening;

    // When non-zero, write a variety of debugging images to the
    // current working directory at various stages through the
    // detection process. (Somewhat slow).
    int debug;

    struct apriltag_quad_thresh_params qtp;

    ///////////////////////////////////////////////////////////////
    // Statistics relating to last processed frame
    timeprofile_t *tp;

    uint32_t nedges;
    uint32_t nsegments;
    uint32_t nquads;

    ///////////////////////////////////////////////////////////////
    // Internal variables below

    // Not freed on apriltag_destroy; a tag family can be shared
    // between multiple users. The user should ultimately destroy the
    // tag family passed into the constructor.

    //zarray_t *tag_families;

    // Used to manage multi-threading.
    //!!!!!workerpool_t *wp;

    // Used for thread safety.
    pthread_mutex_t mutex;

    */
};

typedef struct libtorchtrophallaxis_detection libtorchtrophallaxis_detection_t;
struct libtorchtrophallaxis_detection
{
    /*
    // a pointer for convenience. not freed by apriltag_detection_destroy.
    //apriltag_family_t *family;
    */
    // The decoded ID of the tag
    int id;

    /*
    // How many error bits were corrected? Note: accepting large numbers of
    // corrected errors leads to greatly increased false positive rates.
    // NOTE: As of this implementation, the detector cannot detect tags with
    // a hamming distance greater than 2.
    int hamming;

    // A measure of the quality of the binary decoding process: the
    // average difference between the intensity of a data bit versus
    // the decision threshold. Higher numbers roughly indicate better
    // decodes. This is a reasonable measure of detection accuracy
    // only for very small tags-- not effective for larger tags (where
    // we could have sampled anywhere within a bit cell and still
    // gotten a good detection.)
    float decision_margin;

    // The 3x3 homography matrix describing the projection from an
    // "ideal" tag (with corners at (-1,1), (1,1), (1,-1), and (-1,
    // -1)) to pixels in the image. This matrix will be freed by
    // apriltag_detection_destroy.
    matd_t *H;
    */

    // The center of the detection in image pixel coordinates.
    double c[2];

    // The corners of the tag in image pixel coordinates. These always
    // wrap counter-clock wise around the tag.
    double p[4][2];
};


void libtorchtrophallaxis_detector_destroy(libtorchtrophallaxis_detector_t *td);
/*
{
    /*
    timeprofile_destroy(td->tp);
    workerpool_destroy(td->wp);

    apriltag_detector_clear_families(td);

    zarray_destroy(td->tag_families);
    
    free(td);
}*/

//void testtorch();


libtorchtrophallaxis_detector_t *libtorchtrophallaxis_detector_create();
/*
{
    libtorchtrophallaxis_detector_t *td = (libtorchtrophallaxis_detector_t*) calloc(1, sizeof(libtorchtrophallaxis_detector_t));

    td->nthreads = 1;

    /*
    td->quad_decimate = 2.0;
    td->quad_sigma = 0.0;

    td->qtp.max_nmaxima = 10;
    td->qtp.min_cluster_pixels = 5;

    td->qtp.max_line_fit_mse = 10.0;
    td->qtp.cos_critical_rad = cos(10 * M_PI / 180);
    td->qtp.deglitch = 0;
    td->qtp.min_white_black_diff = 5;

    td->tag_families = zarray_create(sizeof(apriltag_family_t*));

    pthread_mutex_init(&td->mutex, NULL);

    td->tp = timeprofile_create();

    td->refine_edges = 1;
    td->decode_sharpening = 0.25;


    td->debug = 0;

    // NB: defer initialization of td->wp so that the user can
    // override td->nthreads.
    
    return td;
}
*/

#ifdef __cplusplus
}
#endif