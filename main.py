"""
Compare SLIC image segmentation methods using Streamlit.

This program allows the user to adjust the parameters for the SLIC algorithm,
which are used to segment an image into superpixels.
The user can choose between different SLIC methods and adjust parameters.
The resulting segmentation is displayed in real-time using Streamlit.

The program also adds each superpixel segment into an array,
which can be accessed for further analysis or processing.

Author: 0xtommyOfficial (molmez.io)
Date: 2023-03-06
License: This program is released under the MIT License.
"""

import cv2
import time
import numpy as np
import streamlit as st
from PIL import Image
from fast_slic import Slic
from skimage.segmentation import mark_boundaries

# less than 10 can cause fatal memory limit crashes
MINIMUM_REGION_SIZE = 10


def get_super_pixels(image, cluster_map, mark_contours, draw_contours):

    # array of individual super pixels
    superpixels = []

    for (i, seg_val) in enumerate(np.unique(cluster_map)):
        mask = np.zeros(image.shape[:2], dtype='uint8')
        mask[cluster_map == seg_val] = 255

        if (int(cv2.__version__.split(".")[0]) >= 4):
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            im2, contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        superpixel = cv2.bitwise_and(image, image, mask=mask)
        superpixel = cv2.cvtColor(superpixel, cv2.COLOR_BGR2RGB)
        superpixels.append(superpixel)
        if draw_contours:
            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    if mark_contours:
        image = mark_boundaries(image, cluster_map)
        image = (image * 255 / np.max(image)).astype('uint8')

    return image, superpixels


def fast_super_pixel(image, num_components, compactness, min_size_factor):
    new_image = image.copy()
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2LAB)

    t1 = time.perf_counter()

    slic_a = Slic(num_components=num_components, compactness=compactness, min_size_factor=min_size_factor)
    cluster_map = slic_a.iterate(new_image)  # cluster map
    new_image = cv2.cvtColor(new_image, cv2.COLOR_LAB2RGB)
    # print(f'clusters {slic_a.slic_model.clusters}') # The cluster information of superpixels.

    t2 = time.perf_counter()

    new_image, superpixels = get_super_pixels(new_image, cluster_map, False, True)

    t3 = time.perf_counter()
    process_time = round(t2 - t1, 2)
    draw_time = round(t3 - t2, 2)
    return new_image, superpixels, process_time, draw_time


def cv2_super_pixel(image, iterations, region_size):
    new_image = image.copy()

    t1 = time.perf_counter()
    if region_size < MINIMUM_REGION_SIZE:
        region_size = MINIMUM_REGION_SIZE
    slic = cv2.ximgproc.createSuperpixelSLIC(new_image, algorithm=cv2.ximgproc.SLICO, region_size=region_size)
    slic.iterate(iterations)
    cluster_map = slic.getLabels()

    t2 = time.perf_counter()

    new_image, superpixels = get_super_pixels(new_image, cluster_map, False, True)

    t3 = time.perf_counter()
    process_time = round(t2 - t1, 2)
    draw_time = round(t3 - t2, 2)
    return new_image, superpixels, process_time, draw_time


if __name__ == "__main__":

    fast_proc_time = 0
    fast_draw_time = 0
    cv2_proc_time = 0
    cv2_draw_time = 0
    cv2_slic_image = None
    fast_slic_image = None
    image_processed = False

    st.title('SLIC Comparison')
    uploaded_file = st.file_uploader("Upload an image", type=['jpg','png','jpeg'])
    col1, col2 = st.columns(2)

    with col1:
        st.header("cv2 SLICO")
        iterations = st.number_input('Iterations:', value=25, step=1)
        region_size = st.number_input('Region Size:', value=75, step=1)

    with col2:
        st.header("fast-SLIC")
        num_of_components = st.number_input('Number of Components:', value=25, step=1)
        compactness = st.number_input('Compactness:', value=100, step=1)
        min_size_factor = st.number_input('Min Size Factor:', value=0.5, step=0.1)

    empty_col_1, col3, empty_col_2 = st.columns(3)

    with col3:
        if uploaded_file is not None:

            image = Image.open(uploaded_file)
            converted_img = np.array(image.convert('RGB'))
            frame = cv2.resize(converted_img, (512, 512), cv2.INTER_AREA)

            fast_slic_image, fast_slic_superpixels, \
                fast_proc_time, fast_draw_time = fast_super_pixel(image=frame,
                                                                  num_components=num_of_components,
                                                                  compactness=compactness,
                                                                  min_size_factor=min_size_factor)
            cv2_slic_image, cv2_slic_superpixels, \
                cv2_proc_time, cv2_draw_time = cv2_super_pixel(image=frame,
                                                               iterations=iterations,
                                                               region_size=region_size)
            image_processed = True
        else:
            st.write('please upload an image')

    col4, col5 = st.columns(2)

    with col4:
        if cv2_slic_image is not None:
            st.image(cv2_slic_image)

    with col5:
        if fast_slic_image is not None:
            st.image(fast_slic_image)

    col6, col7 = st.columns(2)

    with col6:
        st.header('Performance')
        if image_processed:
            st.write(f'**Processing Time: {cv2_proc_time}**')
            st.write(f'**Drawing Time: {cv2_draw_time}**')
            st.header('Stats')
            st.write(f'Image Size: {cv2_slic_image.shape}')
            st.write(f'Iterations: {iterations}')
            st.write(f'Region Size: {region_size}')

    with col7:
        st.header('Performance')
        if image_processed:
            st.write(f'**Processing Time: {fast_proc_time}**')
            st.write(f'**Drawing Time: {fast_draw_time}**')
            st.header('Stats')
            st.write(f'Image Size: {fast_slic_image.shape}')
            st.write(f'Number Of Components: {num_of_components}')
            st.write(f'Compactness: {compactness}')
            st.write(f'Min Size Factor: {min_size_factor}')
