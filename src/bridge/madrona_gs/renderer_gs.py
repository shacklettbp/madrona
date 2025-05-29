import numpy as np
import torch

from madrona_gs._madrona_gs_batch_renderer import MadronaBatchRenderer
from madrona_gs._madrona_gs_batch_renderer.madrona import ExecMode
from trimesh.visual.texture import TextureVisuals

class BatchRendererGS:
  """Wraps Genesis Model around MadronaBatchRenderer."""

  def __init__(
      self,
      rigid,
      gpu_id,
      num_worlds,
      cameras,
      lights,
      batch_render_view_width=128,
      batch_render_view_height=128,
      add_cam_debug_geo=False,
      use_rasterizer=False,
  ):
    assert rigid is not None, "Rigid body model is required for BatchRendererGS"
    assert gpu_id >= 0, "GPU ID must be greater than or equal to 0"
    assert num_worlds > 0, "Number of worlds must be greater than 0"
    assert batch_render_view_width > 0, "Batch render view width must be greater than 0"
    assert batch_render_view_height > 0, "Batch render view height must be greater than 0"

    # defaults
    default_geom_group = 2
    default_enabled_geom_groups=np.array([default_geom_group])

    self.rigid = rigid
    self.num_worlds = num_worlds
    self.cameras = cameras
    self.lights = lights

    mesh_vertices = rigid.vverts_info.init_pos.to_numpy()
    mesh_faces = rigid.vfaces_info.vverts_idx.to_numpy()
    mesh_vertex_offsets = rigid.vgeoms_info.vvert_start.to_numpy()
    mesh_face_offsets = rigid.vgeoms_info.vface_start.to_numpy()
    n_vgeom = rigid.n_vgeoms

    face_start = rigid.vgeoms_info.vface_start.to_numpy()
    face_end = rigid.vgeoms_info.vface_end.to_numpy()
    for i in range(n_vgeom):
      mesh_faces[face_start[i] : face_end[i]] -= mesh_vertex_offsets[i]

    geom_types = np.full((n_vgeom,), 7, dtype=np.int32)
    geom_groups = np.full((n_vgeom,), default_geom_group, dtype=np.int32)
    geom_data_ids = np.arange(n_vgeom, dtype=np.int32)
    geom_sizes = np.ones((n_vgeom, 3), dtype=np.float32)
    num_cams = len(cameras) if cameras is not None else 0
    assert num_cams > 0, "Must have at least one camera for Madrona to work!"
    num_lights = len(lights) if lights is not None else 0
    geom_rgba = rigid.vgeoms_info.color.to_numpy()

    geom_mat_ids, mesh_texcoord_num, mesh_texcoord_offsets, mesh_texcoord_data, texture_widths, texture_heights, texture_nchans, texture_data, texture_offsets, material_texture_ids, material_rgba = self.get_texture_data()

    # TODO: Support mutable camera fov
    cam_fovy = np.array([cam.fov for cam in cameras], dtype=np.float32)#

    self.madrona = MadronaBatchRenderer(
        gpu_id=gpu_id,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        mesh_vertex_offsets=mesh_vertex_offsets,
        mesh_face_offsets=mesh_face_offsets,
        mesh_texcoords=mesh_texcoord_data,
        mesh_texcoord_offsets=mesh_texcoord_offsets,
        mesh_texcoord_num=mesh_texcoord_num,
        geom_types=geom_types,
        geom_groups=geom_groups,
        geom_data_ids=geom_data_ids,
        geom_sizes=geom_sizes,
        geom_mat_ids=geom_mat_ids,
        geom_rgba=geom_rgba,
        mat_rgba=material_rgba,
        mat_tex_ids=material_texture_ids,

        tex_data=texture_data,
        tex_offsets=texture_offsets,
        tex_widths=texture_widths,
        tex_heights=texture_heights,
        tex_nchans=texture_nchans,
        num_lights=num_lights,
        num_cams=num_cams,
        num_worlds=num_worlds,

        batch_render_view_width=batch_render_view_width,
        batch_render_view_height=batch_render_view_height,
        cam_fovy=cam_fovy,
        enabled_geom_groups=default_enabled_geom_groups,
        add_cam_debug_geo=add_cam_debug_geo,
        use_rt=not use_rasterizer,
    )
    

  def init(self):
    #print("init")
    #print("cam_pos", cam_pos)
    #print("cam_rot", cam_rot) 
    #print("geom_pos", geom_pos)
    #print("geom_rot", geom_rot)
    #print("geom_mat_ids", geom_mat_ids)
    #print("geom_rgba", geom_rgba)
    #print("geom_sizes", geom_sizes)
    #print("light_pos", light_pos, light_pos.shape)
    #print("light_dir", light_dir, light_dir.shape)
    #print("light_directional", light_directional, light_directional.shape)
    #print("light_castshadow", light_castshadow, light_castshadow.shape)
    #print("light_cutoff", light_cutoff, light_cutoff.shape)

    cam_pos, cam_rot = self.get_camera_pos_rot_numpy()
    geom_pos, geom_rot = self.get_geom_pos_rot_numpy()

    geom_mat_ids = np.full((self.rigid.n_vgeoms,), -1, dtype=np.int32)
    geom_rgba = self.rigid.vgeoms_info.color.to_numpy()
    geom_rgba_uint = np.array(geom_rgba * 255, np.uint32) 
    geom_rgb = geom_rgba_uint[...,0] * (1 << 16) + geom_rgba_uint[...,1] * (1 << 8) + geom_rgba_uint[...,2]
    geom_sizes = np.ones((self.rigid.n_vgeoms, 3), dtype=np.float32)#

    light_pos, light_dir, light_directional, light_castshadow, light_cutoff, light_intensity = self.get_lights_properties_numpy()

    # Make a copy to actually shuffle the memory layout before passing to C++
    self.madrona.init(
      geom_pos.copy(),
      geom_rot.copy(),
      cam_pos.copy(),
      cam_rot.copy(),
      np.repeat(geom_mat_ids[np.newaxis], self.num_worlds, axis=0),
      np.repeat(geom_rgb[np.newaxis], self.num_worlds, axis=0),
      np.repeat(geom_sizes[np.newaxis], self.num_worlds, axis=0),
      np.repeat(light_pos[np.newaxis], self.num_worlds, axis=0),
      np.repeat(light_dir[np.newaxis], self.num_worlds, axis=0),
      np.repeat(light_directional[np.newaxis], self.num_worlds, axis=0),
      np.repeat(light_castshadow[np.newaxis], self.num_worlds, axis=0),
      np.repeat(light_cutoff[np.newaxis], self.num_worlds, axis=0),
      np.repeat(light_intensity[np.newaxis], self.num_worlds, axis=0),
    )


  def render(self):
    #print("here is rendering")      
    # Assume execution on GPU
    # TODO: Need to check if the device is GPU or CPU, or assert if not GPU
    cam_pos, cam_rot = self.get_camera_pos_rot_torch()
    geom_pos, geom_rot = self.get_geom_pos_rot_torch()
    #print("geom_pos", geom_pos.shape, geom_pos.dtype, geom_pos.device)
    #print("geom_rot", geom_rot.shape, geom_rot.dtype, geom_rot.device)
    #print("cam_pos", cam_pos.shape, cam_pos.dtype, cam_pos.device)
    #print("cam_rot", cam_rot.shape, cam_rot.dtype, cam_rot.device)

    #self.madrona.render_dummy()
    self.madrona.render_torch(
        geom_pos,
        geom_rot,
        cam_pos,
        cam_rot,
    )
    rgb_torch = self.madrona.rgb_tensor().to_torch()
    depth_torch = self.madrona.depth_tensor().to_torch()    
    return rgb_torch, depth_torch
  
  def destroy(self):
    self.rigid = None
    self.cameras = None
    self.lights = None

  def get_texture_data(self):
    n_vgeom = self.rigid.n_vgeoms
    vgeoms = self.rigid.vgeoms

    # get number of textures, total texcoord and texture data size
    num_textures = 0
    total_texcoord_data_size = 0
    total_texture_data_size = 0
    for geomIdx, vgeom in enumerate(vgeoms):
      visual = vgeom.get_trimesh().visual
      if(isinstance(visual, TextureVisuals)):
        total_texcoord_data_size += visual.uv.shape[0]
        texture_width = visual.material.image.width
        texture_height = visual.material.image.height
        texture_nchans = 4 if visual.material.image.mode == "RGBA" else 3
        texture_data_size = texture_width * texture_height * texture_nchans
        total_texture_data_size += texture_data_size
        num_textures += 1

    # allocate memory for texcoord data
    geom_mat_ids = np.full(n_vgeom, -1, dtype=np.int32)
    mesh_texcoord_num = np.full(n_vgeom, 0, dtype=np.int32)
    mesh_texcoord_offsets = np.full(n_vgeom, -1, dtype=np.int32)
    texcoord_data = np.empty((total_texcoord_data_size, 2), dtype=np.float32)
    texture_widths = np.empty(num_textures, dtype=np.int32)
    texture_heights = np.empty(num_textures, dtype=np.int32)
    texture_nchans = np.empty(num_textures, dtype=np.int32)
    texture_data = np.empty(total_texture_data_size, dtype=np.uint8)
    texture_offsets = np.empty(num_textures, dtype=np.int32)
    num_textures_per_material = 10 # Madrona allows up to 10 textures per material
    material_texture_ids = np.full((num_textures, num_textures_per_material), -1, dtype=np.int32)
    material_rgba = np.empty((num_textures, 4), dtype=np.float32)

    num_textures = 0 # reset index
    total_texcoord_data_size = 0 # reset size
    total_texture_data_size = 0 # reset size
    for geomIdx, vgeom in enumerate(vgeoms):
      visual = vgeom.get_trimesh().visual
      if(isinstance(visual, TextureVisuals)):
        # Copy texcoord data
        uv_size = visual.uv.shape
        mesh_texcoord_num[num_textures] = uv_size[0]
        mesh_texcoord_offsets[num_textures] = total_texcoord_data_size
        total_texcoord_data_size += uv_size[0] * 2
        texcoord_data[mesh_texcoord_offsets[num_textures] : mesh_texcoord_offsets[num_textures] + mesh_texcoord_num[num_textures] * 2] = visual.uv.astype(np.float32)

        # Copy texture data
        texture_widths[num_textures] = visual.material.image.width
        texture_heights[num_textures] = visual.material.image.height
        texture_nchans[num_textures] = 4 if visual.material.image.mode == "RGBA" else 3
        texture_data_size = texture_widths[num_textures] * texture_heights[num_textures] * texture_nchans[num_textures]
        texture_offsets[num_textures] = total_texture_data_size
        total_texture_data_size += texture_data_size
        texture_data[texture_offsets[num_textures] : texture_offsets[num_textures] + texture_data_size] = np.array(list(visual.material.image.getdata()), dtype=np.uint8).flatten()

        # Set material id
        geom_mat_ids[geomIdx] = num_textures
        material_texture_ids[num_textures, 0] = num_textures # Use first texture as diffuse
        material_rgba[num_textures] = visual.material.diffuse.astype(np.float32) / 255.0

        # Bump texture index
        num_textures += 1

    return geom_mat_ids, mesh_texcoord_num, mesh_texcoord_offsets, texcoord_data, texture_widths, texture_heights, texture_nchans, texture_data, texture_offsets, material_texture_ids, material_rgba

########################## Utils ##########################
  def get_camera_pos_rot_numpy(self):
    cam_pos = np.array([c.pos for c in self.cameras], dtype=np.float32)
    cam_rot = np.array([c.quat_for_madrona for c in self.cameras], dtype=np.float32)
    cam_pos = np.repeat(cam_pos[None], self.num_worlds, axis=0)
    cam_rot = np.repeat(cam_rot[None], self.num_worlds, axis=0)
    return cam_pos, cam_rot
  
  def get_camera_pos_rot_torch(self):
    cam_pos, cam_rot = self.get_camera_pos_rot_numpy()
    cam_pos = torch.tensor(cam_pos).to("cuda")
    cam_rot = torch.tensor(cam_rot).to("cuda")
    return cam_pos, cam_rot
  
  def get_geom_pos_rot_numpy(self):
    geom_pos = self.rigid.vgeoms_state.pos.to_numpy()
    geom_rot = self.rigid.vgeoms_state.quat.to_numpy()
    geom_pos = np.swapaxes(geom_pos, 0, 1)
    geom_rot = np.swapaxes(geom_rot, 0, 1)
    return geom_pos, geom_rot
  
  def get_geom_pos_rot_torch(self):
    geom_pos = self.rigid.vgeoms_state.pos.to_torch()
    geom_rot = self.rigid.vgeoms_state.quat.to_torch()
    geom_pos = geom_pos.transpose(0, 1).contiguous().to("cuda")
    geom_rot = geom_rot.transpose(0, 1).contiguous().to("cuda")
    return geom_pos, geom_rot
  
  def get_lights_properties_numpy(self):
    light_pos = np.array([l.pos for l in self.lights], dtype=np.float32).reshape(-1, 3)
    light_dir = np.array([l.dir for l in self.lights], dtype=np.float32).reshape(-1, 3)
    light_directional = np.array([l.directional for l in self.lights], dtype=np.uint8).reshape(-1)
    light_castshadow = np.array([l.castshadow for l in self.lights], dtype=np.uint8).reshape(-1)
    light_cutoff = np.array([l.cutoffRad for l in self.lights], dtype=np.float32).reshape(-1)
    light_intensity = np.array([l.intensity for l in self.lights], dtype=np.float32).reshape(-1)
    return light_pos, light_dir, light_directional, light_castshadow, light_cutoff, light_intensity
