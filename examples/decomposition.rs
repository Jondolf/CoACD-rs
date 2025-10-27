use bevy::{
    asset::RenderAssetUsages,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    light::{DirectionalLightShadowMap, NotShadowCaster},
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
    scene::SceneInstanceReady,
};
use bevy_obj::ObjPlugin;
use coacd::{Coacd, mesh::IndexedMesh, parameters::CoacdParaneters};
use rand::{Rng, rng};

fn main() {
    let mut app = App::new();

    // Add some plugins.
    app.add_plugins((DefaultPlugins, ObjPlugin));

    // Disable shadow casting for all meshes.
    app.register_required_components::<Mesh3d, NotShadowCaster>();

    // Increase shadow map resolution for better quality shadows.
    app.insert_resource(DirectionalLightShadowMap { size: 4096 });

    // Add the convex decomposition observer and systems.
    app.add_observer(spawn_convex_decomposition)
        .add_systems(Startup, setup)
        .add_systems(Update, rotate)
        .run();
}

#[derive(Component)]
struct Rotating;

#[derive(Component)]
struct SourceMesh;

fn setup(mut commands: Commands, assets: ResMut<AssetServer>) {
    // Spawn some 3D models to decompose.
    commands.spawn((
        SceneRoot(assets.load("SquareRing.obj")),
        Transform::from_xyz(-4.25, 1.5, 0.0).with_scale(Vec3::splat(0.75)),
        SourceMesh,
        Rotating,
    ));
    commands.spawn((
        SceneRoot(assets.load("SnowFlake.obj")),
        Transform::from_xyz(0.0, 1.5, 0.0).with_scale(Vec3::splat(0.03)),
        SourceMesh,
        Rotating,
    ));
    commands.spawn((
        SceneRoot(assets.load("Octocat-v2.obj")),
        Transform::from_xyz(4.25, 1.0, 0.0),
        SourceMesh,
        Rotating,
    ));

    // Directional light
    commands.spawn((
        DirectionalLight {
            illuminance: 60_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::default().looking_at(Vec3::new(-1.0, -2.5, -1.5), Vec3::Y),
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        AmbientLight {
            brightness: 10_000.0,
            color: Color::WHITE,
            ..default()
        },
        Exposure::SUNLIGHT,
        Tonemapping::AcesFitted,
        Transform::from_xyz(0.0, 1.5, 10.0).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
    ));
}

fn rotate(mut query: Query<&mut Transform, With<Rotating>>, time: Res<Time>) {
    let angle = time.elapsed_secs() * core::f32::consts::PI / 4.0;
    for mut transform in &mut query {
        transform.rotation = Quat::from_rotation_y(angle);
    }
}

fn spawn_convex_decomposition(
    ready: On<SceneInstanceReady>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    child_query: Query<&Children>,
    mesh_query: Query<(&Mesh3d, &MeshMaterial3d<StandardMaterial>)>,
    transform_query: Query<&Transform>,
) {
    let scene_transform = transform_query.get(ready.entity).unwrap();

    for (mesh, material) in mesh_query.iter_many(child_query.iter_descendants(ready.entity)) {
        // Set the material of the source mesh to be white.
        let material = materials.get_mut(material.id()).unwrap();
        material.base_color = Color::WHITE;

        // Extract the mesh data.
        let mesh = meshes.get(mesh.id()).unwrap();
        let vertex_positions = mesh.attribute(Mesh::ATTRIBUTE_POSITION).unwrap();
        let vertex_positions = vertex_positions.as_float3().unwrap();

        // Create an indexed mesh for CoACD.
        let vertices = vertex_positions
            .iter()
            .map(|&v| vec3a(v[0], v[1], v[2]))
            .collect::<Vec<_>>();
        let indices: Vec<[usize; 3]> = mesh
            .indices()
            .unwrap()
            .iter()
            .collect::<Vec<_>>()
            .chunks(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect();
        let indexed_mesh = IndexedMesh { vertices, indices };

        // Decompose the mesh using CoACD.
        let coacd = Coacd {
            parameters: CoacdParaneters::new(0.03),
        };
        let parts = coacd.decompose(&indexed_mesh);

        // Spawn each convex hull part.
        for part in parts {
            let mesh_vertices: Vec<[f32; 3]> =
                part.vertices.iter().map(|v| [v.x, v.y, v.z]).collect();
            let mesh_indices: Vec<u32> = part
                .indices
                .iter()
                .flat_map(|tri| tri.iter().map(|&i| i as u32))
                .collect();

            // Create a mesh from the hull.
            let hull_mesh = Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, mesh_vertices)
            .with_inserted_indices(Indices::U32(mesh_indices))
            .with_duplicated_vertices()
            .with_computed_flat_normals();

            let random_hue: f32 = rng().random_range(0.0..360.0);

            // Spawn the hull mesh.
            commands.spawn((
                Mesh3d(meshes.add(hull_mesh)),
                MeshMaterial3d(materials.add(StandardMaterial {
                    base_color: Color::hsl(random_hue, 1.0, 0.75),
                    ..default()
                })),
                Transform::from_xyz(
                    scene_transform.translation.x,
                    scene_transform.translation.y - 3.0,
                    scene_transform.translation.z,
                )
                .with_scale(scene_transform.scale),
                Rotating,
            ));
        }
    }
}
