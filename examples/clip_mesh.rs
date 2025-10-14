use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

/// set up a simple 3D scene
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = Mesh::from(ConicalFrustum {
        radius_top: 1.0,
        radius_bottom: 2.0,
        height: 2.0,
    });

    let indexed_mesh = coacd::mesh::IndexedMesh {
        vertices: mesh
            .attribute(Mesh::ATTRIBUTE_POSITION)
            .unwrap()
            .as_float3()
            .unwrap()
            .iter()
            .map(|&[x, y, z]| Vec3A::new(x, y, z))
            .collect(),
        indices: match mesh.indices().unwrap() {
            Indices::U16(indices) => indices
                .chunks(3)
                .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
                .collect(),
            Indices::U32(indices) => indices
                .chunks(3)
                .map(|chunk| [chunk[0] as usize, chunk[1] as usize, chunk[2] as usize])
                .collect(),
        },
    };

    let result = coacd::clip::clip(
        &indexed_mesh,
        &coacd::Plane::new(Vec3A::new(-1.0, 1.0, 0.0).normalize(), 0.0),
    )
    .unwrap();

    let positive_vertices: Vec<[f32; 3]> = result
        .positive_mesh
        .vertices
        .iter()
        .map(|v| [v.x, v.y, v.z])
        .collect();
    let positive_indices: Vec<u32> = result
        .positive_mesh
        .indices
        .iter()
        .flat_map(|i| i.iter().map(|&idx| idx as u32))
        .collect();
    let mut positive_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positive_vertices)
    .with_inserted_indices(Indices::U32(positive_indices));
    positive_mesh.duplicate_vertices();
    positive_mesh.compute_flat_normals();

    let negative_vertices: Vec<[f32; 3]> = result
        .negative_mesh
        .vertices
        .iter()
        .map(|v| [v.x, v.y, v.z])
        .collect();
    let negative_indices: Vec<u32> = result
        .negative_mesh
        .indices
        .iter()
        .flat_map(|i| i.iter().map(|&idx| idx as u32))
        .collect();
    let mut negative_mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, negative_vertices)
    .with_inserted_indices(Indices::U32(negative_indices));
    negative_mesh.duplicate_vertices();
    negative_mesh.compute_flat_normals();

    // cube
    commands.spawn((
        Mesh3d(meshes.add(positive_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            cull_mode: None,
            base_color: Color::srgb_u8(255, 50, 50),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
    commands.spawn((
        Mesh3d(meshes.add(negative_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            cull_mode: None,
            base_color: Color::srgb_u8(50, 50, 255),
            ..default()
        })),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // light
    commands.spawn((
        PointLight {
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
