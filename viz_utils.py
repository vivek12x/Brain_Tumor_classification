# viz_utils.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from nilearn import datasets, surface

def make_overlay(orig_rgb, cam_norm, alpha=0.45):
    """
    orig_rgb numpy HxWx3 uint8, cam_norm float HxW 0..1
    returns PIL image overlay
    """
    import cv2
    heat = (255 * cam_norm).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # BGR
    heat = heat[..., ::-1]  # to RGB
    blended = cv2.addWeighted(orig_rgb.astype(np.uint8), 1-alpha, heat, alpha, 0)
    return Image.fromarray(blended)

def save_image(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)

def project_centroid_to_fsaverage(centroid, image_size, cam):
    """
    Rough mapping of 2D centroid to template brain mesh vertex highlight.
    centroid: (x,y) in pixel coords on resized image
    image_size: width or height (square)
    cam: 2D cam array HxW used to weight which hemisphere to highlight

    Returns:
      mesh: (coords, faces) tuple of the hemisphere chosen
      vert_color: numpy array length == coords.shape[0] with highlight weights
    """
    # fetch fsaverage mesh (no subjects_dir kw)
    fs = datasets.fetch_surf_fsaverage()
    mesh_left = surface.load_surf_mesh(fs['pial_left'])
    mesh_right = surface.load_surf_mesh(fs['pial_right'])

    # compute normalized centroid in [0,1]
    x, y = centroid
    nx = x / float(image_size)
    ny = y / float(image_size)

    # choose hemisphere by nx
    hemi = 'left' if nx < 0.5 else 'right'

    if hemi == 'left':
        coords, faces = mesh_left
    else:
        coords, faces = mesh_right

    # map normalized coords to a 3D unit vector (very rough)
    theta = nx * 2 * np.pi
    phi = (ny - 0.5) * np.pi
    v = np.array([
        np.cos(phi) * np.cos(theta),
        np.cos(phi) * np.sin(theta),
        np.sin(phi)
    ])

    # find nearest vertex
    dists = np.sqrt(((coords - v[None,:])**2).sum(axis=1))
    vidx = int(dists.argmin())

    # build vertex color arrays: highlight a small neighborhood
    vert_color = np.zeros((coords.shape[0],), dtype=float)
    vert_color[vidx] = 1.0

    # expand to local neighbors using faces adjacency
    neighbors = set()
    for f in faces:
        if vidx in f:
            for vtx in f:
                neighbors.add(int(vtx))
    for n in neighbors:
        vert_color[n] = max(vert_color[n], 0.6)

    return (coords, faces), vert_color

def plot_fsaverage_highlight(mesh, vert_color):
    """
    Create a plotly figure of a hemisphere mesh with highlighted vertices.
    mesh: (coords, faces)
    vert_color: array matching vertices length
    """
    coords, faces = mesh
    x, y, z = coords.T
    i, j, k = faces.T

    mesh3d = go.Mesh3d(x=x, y=y, z=z,
                       i=i, j=j, k=k,
                       intensity=vert_color,
                       colorscale='Hot',
                       showscale=False,
                       opacity=0.9)

    fig = go.Figure(data=[mesh3d])
    fig.update_layout(scene=dict(aspectmode='data'), title="fsaverage brain highlight demo")
    return fig
