#include "trophallaxis.hpp"
//#include <torch/torch.h>

#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include <iostream>

void libtorchtrophallaxis_detector_destroy(libtorchtrophallaxis_detector_t *td)
{
    /*
    timeprofile_destroy(td->tp);
    workerpool_destroy(td->wp);

    apriltag_detector_clear_families(td);

    zarray_destroy(td->tag_families);
    */
    free(td);
}

libtorchtrophallaxis_detector_t *libtorchtrophallaxis_detector_create()
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
    */
    return td;
}

/*
void testtorch()
{
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;
}*/

