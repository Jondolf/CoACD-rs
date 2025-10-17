//! [Monte Carlo Tree Search (MCTS)][MCTS] for optimal splitting plane selection.
//!
//! [MCTS]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

#![expect(missing_docs)]

use glam::Vec3A;
use obvhs::aabb::Aabb;

use crate::{
    Plane, cost,
    mesh::{ConvexHullExt, IndexedMesh},
    parameters::CoacdParaneters,
};

pub const MCTS_RANDOM_CUT: u32 = 1;

/// A Monte Carlo tree for finding optimal splitting planes using
/// [Monte Carlo Tree Search (MCTS)][MCTS].
///
/// [MCTS]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
pub struct MonteCarloTree {
    /// All nodes in the Monte Carlo tree.
    nodes: Vec<Node>,
}

impl MonteCarloTree {
    /// Creates a new [`MonteCarloTree`] with the given initial mesh and parameters.
    #[inline]
    pub fn new(mesh: IndexedMesh, parameters: &CoacdParaneters) -> Self {
        let root_node = Node::new(NodeState::new(mesh, parameters));
        Self {
            nodes: vec![root_node],
        }
    }

    /// Clears the tree, removing all nodes.
    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
    }

    /// Returns a reference to the node at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    #[inline]
    pub fn node(&self, index: NodeIndex) -> &Node {
        &self.nodes[index.0 as usize]
    }

    /// Adds a new node with the given state and parent index to the tree.
    #[inline]
    pub fn add_node_with_parent(&mut self, parent: NodeIndex, state: NodeState) -> NodeIndex {
        // Create a new node with the given state and parent index.
        let new_index = NodeIndex(self.nodes.len() as u32);
        let mut new_node = Node::new(state);
        new_node.parent = Some(parent);
        self.nodes.push(new_node);

        // Add the new node as a child of the parent node.
        if let Some(parent_node) = self.nodes.get_mut(parent.0 as usize) {
            parent_node.children.push(new_index);
        }

        new_index
    }

    /// Performs [Monte Carlo Tree Search (MCTS)][MCTS] to find the optimal splitting plane.
    ///
    /// [MCTS]: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
    pub fn search(&mut self, parameters: &CoacdParaneters) -> Option<Plane> {
        // Start the MCTS process from the root node.
        let root_index = NodeIndex(0);
        let mut best_path = Vec::new();

        // Precompute the initial cost for normalization in UCB.
        let initial_mesh = &self.nodes[root_index.0 as usize].state.initial_part;
        let initial_hull = initial_mesh
            .compute_convex_hull()
            .expect("Failed to compute convex hull")
            .to_mesh();
        let initial_cost = parameters.rv_k * cost::compute_rv(initial_mesh, &initial_hull)
            / parameters.mcts_max_depth as f32;

        // Store the current path of planes during simulation.
        let mut current_path = Vec::new();

        // Perform MCTS iterations.
        for _ in 0..parameters.mcts_iterations {
            // Select a node to expand using the tree policy.
            let selected_index = self.tree_policy(initial_cost, parameters);

            // Simulate a random playout from the selected node.
            let reward = self.default_policy(selected_index, &mut current_path, parameters);

            // Backpropagate the reward up the tree.
            self.backpropagate(selected_index, reward, &current_path, &mut best_path);

            // Clear the current path for the next iteration.
            current_path.clear();
        }

        // Select the best child of the root node as the final result.
        let best_child_index = self.best_child(root_index, false, initial_cost)?;

        // Get the best node and its associated plane.
        let best_node = &self.nodes[best_child_index.0 as usize];
        let mut plane = best_node.state.current_value.expect("No plane found").0;

        // Refine the best plane using ternary search.
        ternary_mcts(
            &self.nodes[root_index.0 as usize].state.initial_part,
            &mut plane,
            &best_path,
            best_node.quality_value,
            1e-4,
            parameters,
        );

        Some(plane)
    }

    /// Expands the given node by adding a new child node with a random move.
    fn expand_node(&mut self, node_index: NodeIndex, parameters: &CoacdParaneters) -> NodeIndex {
        let node = &mut self.nodes[node_index.0 as usize];

        // Create the next state by performing a random move.
        if let Some(next_state) = node.state.get_next_state_with_random_choice(parameters) {
            // Add the new node to the tree and return its index.
            self.add_node_with_parent(node_index, next_state)
        } else {
            node_index
        }
    }

    fn default_policy(
        &mut self,
        node_index: NodeIndex,
        current_path: &mut Vec<Plane>,
        parameters: &CoacdParaneters,
    ) -> f32 {
        let current_state = &mut self.nodes[node_index.0 as usize].state;

        // Store planes here to avoid reallocating every iteration.
        let mut planes = Vec::new();

        // Simulate until a terminal state is reached.
        while !current_state.is_terminal(parameters) {
            let worst_part = &current_state.current_parts[current_state.worst_part_index];

            // Compute axis-aligned candidate planes for the worst part.
            compute_axis_aligned_planes(worst_part.current_aabb, MCTS_RANDOM_CUT, &mut planes);
            if planes.is_empty() {
                break;
            }

            // Find the best plane among the candidates.
            let Some(best_plane) =
                find_best_rv_plane(&worst_part.current_mesh, &planes, parameters)
            else {
                break;
            };
            current_path.push(best_plane);

            // Clip the worst part with the selected plane.
            let clip_result = crate::clip::clip(&worst_part.current_mesh, &best_plane)
                .expect("Wrong MCTS plane found");

            // TODO: Can we avoid converting to a mesh here?
            let positive_hull = clip_result
                .positive_mesh
                .compute_convex_hull()
                .expect("Failed to compute convex hull")
                .to_mesh();
            let negative_hull = clip_result
                .negative_mesh
                .compute_convex_hull()
                .expect("Failed to compute convex hull")
                .to_mesh();

            // Remove the worst part from the current state.
            current_state
                .current_parts
                .remove(current_state.worst_part_index);
            current_state
                .current_costs
                .remove(current_state.worst_part_index);

            // Compute costs for the two new parts created by the split.
            let positive_cost =
                parameters.rv_k * cost::compute_rv(&clip_result.positive_mesh, &positive_hull);
            let negative_cost =
                parameters.rv_k * cost::compute_rv(&clip_result.negative_mesh, &negative_hull);
            current_state.current_costs.push(positive_cost);
            current_state.current_costs.push(negative_cost);

            // Create new parts for the two new meshes.
            let positive_part = Part::new(clip_result.positive_mesh, parameters);
            let negative_part = Part::new(clip_result.negative_mesh, parameters);
            current_state.current_parts.push(positive_part);
            current_state.current_parts.push(negative_part);

            // Update the current state.
            current_state.update_score();
            current_state.current_cost += current_state.current_score;
            current_state.current_round += 1;

            // Clear planes for the next iteration.
            planes.clear();
        }

        // Return the normalized cost as the reward.
        current_state.current_cost / parameters.mcts_max_depth as f32
    }

    /// Performs the tree policy to select a node for expansion.
    ///
    /// This involves traversing the tree using the UCB formula until a node
    /// that is not fully expanded or is terminal is found.
    fn tree_policy(&mut self, initial_cost: f32, parameters: &CoacdParaneters) -> NodeIndex {
        // Start from the root node.
        let mut current_index = NodeIndex(0);

        loop {
            let current_node = &self.nodes[current_index.0 as usize];

            // If the current node is terminal, return it.
            if current_node.state.is_terminal(parameters) {
                return current_index;
            }

            // If the current node is not fully expanded, expand it.
            if !current_node.is_fully_expanded() {
                return self.expand_node(current_index, parameters);
            }

            // Otherwise, move to the best child node.
            if let Some(best_child) = self.best_child(current_index, true, initial_cost) {
                current_index = best_child;
            }
        }
    }

    /// Selects the best child node based on the Upper Confidence Bound (UCB) formula.
    ///
    /// If the node has no children, `None` is returned.
    fn best_child(
        &self,
        node_index: NodeIndex,
        explore: bool,
        initial_cost: f32,
    ) -> Option<NodeIndex> {
        let node = &self.nodes[node_index.0 as usize];
        let mut best_index = None;
        let mut best_value = f32::MAX;

        for &child_index in &node.children {
            let child = &self.nodes[child_index.0 as usize];

            // UCB formula: Q + c * sqrt(2 * ln(N) / n)

            // Exploitation term, the average reward
            let exploitation = child.quality_value;

            // Exploration term, encourages exploring less-visited nodes
            let exploration = if explore {
                let c = initial_cost * core::f32::consts::FRAC_1_SQRT_2;
                c * (2.0 * (node.visit_times as f32).ln() / (child.visit_times as f32)).sqrt()
            } else {
                0.0
            };

            // We negate the exploration term because lower cost is better in this context.
            let ucb_value = exploitation - exploration;

            if ucb_value < best_value {
                best_value = ucb_value;
                best_index = Some(child_index);
            }
        }

        best_index
    }

    /// Backpropagates the reward up the tree, updating visit counts and quality values.
    fn backpropagate(
        &mut self,
        node_index: NodeIndex,
        reward: f32,
        current_path: &[Plane],
        best_path: &mut Vec<Plane>,
    ) {
        // Temporary path to store the current path during traversal.
        let mut temp_path = current_path.iter().rev().cloned().collect::<Vec<Plane>>();

        let mut current_index = Some(node_index);

        // Traverse up the tree to the root, updating visit counts and quality values.
        while let Some(index) = current_index {
            let node = &mut self.nodes[index.0 as usize];

            // If we're at the root node and the current reward
            // is better than the best found so far, update the best path.
            if node.state.current_round == 0 && node.quality_value > reward {
                *best_path = temp_path.clone();
            }

            if let Some((plane, _)) = node.state.current_value {
                temp_path.push(plane);
            }

            node.visit_times += 1;
            node.quality_value = node.quality_value.min(reward);

            current_index = node.parent;
        }
    }
}

/// The index of a node in the [`MonteCarloTree`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeIndex(pub u32);

pub struct Node {
    parent: Option<NodeIndex>,
    children: Vec<NodeIndex>,
    state: NodeState,
    visit_times: u32,
    quality_value: f32,
}

impl Node {
    /// Creates a new [`Node`] with the given index and state.
    #[inline]
    const fn new(state: NodeState) -> Self {
        Self {
            children: Vec::new(),
            state,
            visit_times: 0,
            // TODO: Is this the right initial value?
            quality_value: f32::MAX,
            parent: None,
        }
    }

    /// Checks if the node is fully expanded, meaning that all possible child nodes have been created.
    #[inline]
    fn is_fully_expanded(&self) -> bool {
        let current_max_expansions = self.state.current_parts[self.state.worst_part_index]
            .candidate_planes
            .len();
        self.children.len() == current_max_expansions
    }
}

/// The state of a [`Node`] in the [`MonteCarloTree`].
pub struct NodeState {
    pub current_value: Option<(Plane, u32)>,
    /// Current accumulated score.
    // TODO: Do we need to store all of these? Some might just be computed on-the-fly.
    pub current_cost: f32,
    pub current_score: f32,
    pub current_round: u32,
    pub initial_part: IndexedMesh,
    pub ori_mesh_area: f32,
    pub ori_mesh_volume: f32,
    pub ori_hull_volume: f32,
    /// Current costs for each part in `current_parts`.
    pub current_costs: Vec<f32>,
    pub current_parts: Vec<Part>,
    pub worst_part_index: usize,
}

impl NodeState {
    pub fn new(initial_part: IndexedMesh, parameters: &CoacdParaneters) -> Self {
        Self::from_parts(
            initial_part.clone(),
            vec![f32::MAX],
            vec![Part::new(initial_part, parameters)],
        )
    }

    pub fn from_parts(
        initial_part: IndexedMesh,
        current_costs: Vec<f32>,
        current_parts: Vec<Part>,
    ) -> Self {
        let ori_mesh_area = initial_part.surface_area();
        let ori_mesh_volume = initial_part.signed_volume();
        let ori_hull_volume = initial_part
            .compute_convex_hull()
            .expect("Failed to compute convex hull")
            .volume() as f32;

        Self {
            current_value: None,
            current_cost: 0.0,
            current_score: f32::MAX,
            current_round: 0,
            initial_part,
            ori_mesh_area,
            ori_mesh_volume,
            ori_hull_volume,
            current_costs,
            current_parts,
            worst_part_index: 0,
        }
    }

    /// Checks if the current state is terminal based on the given parameters.
    #[inline]
    pub fn is_terminal(&self, parameters: &CoacdParaneters) -> bool {
        self.current_round >= parameters.mcts_max_depth
            || self.current_parts[self.worst_part_index]
                .candidate_planes
                .is_empty()
    }

    pub fn update_score(&mut self) {
        let mut reward = 0.0;
        let mut max_cost = 0.0;

        for (i, &cost) in self.current_costs.iter().enumerate() {
            if cost > max_cost {
                max_cost = cost;
                self.worst_part_index = i;
            }
            reward += cost;
        }

        // TODO: Wut???  The C++ implementation returns the max cost (like we do currently)
        //       and ignores the reward variable entirely. Is this a bug or intentional?
        self.current_score = max_cost;
    }

    /// Performs a move by taking the next available plane from the worst part.
    ///
    /// # Panics
    ///
    /// Panics if there are no more candidate planes available in the worst part.
    #[inline]
    pub fn perform_next_move(&mut self) -> Plane {
        self.current_parts[self.worst_part_index].take_next_move()
    }

    pub fn get_next_state_with_random_choice(
        &mut self,
        parameters: &CoacdParaneters,
    ) -> Option<NodeState> {
        let plane = self.perform_next_move();

        let worst_part = &self.current_parts[self.worst_part_index];

        // Clip the worst part with the selected plane.
        if let Some(clip_result) = crate::clip::clip(&worst_part.current_mesh, &plane) {
            // TODO: Can we avoid converting to a mesh here?
            // TODO: These should never fail ideally. But if they do, we should handle it more gracefully.
            let positive_hull = clip_result
                .positive_mesh
                .compute_convex_hull()
                .ok()?
                .to_mesh();
            let negative_hull = clip_result
                .negative_mesh
                .compute_convex_hull()
                .ok()?
                .to_mesh();

            let mut current_costs = Vec::with_capacity(self.current_costs.len());
            let mut current_parts = Vec::with_capacity(self.current_parts.len());

            // Retain all parts except the worst part, which is being split.
            for i in 0..self.current_parts.len() {
                if i != self.worst_part_index {
                    current_costs.push(self.current_costs[i]);
                    current_parts.push(self.current_parts[i].clone());
                }
            }

            // Compute costs for the two new parts created by the split.
            let positive_cost =
                parameters.rv_k * cost::compute_rv(&clip_result.positive_mesh, &positive_hull);
            let negative_cost =
                parameters.rv_k * cost::compute_rv(&clip_result.negative_mesh, &negative_hull);
            current_costs.push(positive_cost);
            current_costs.push(negative_cost);

            // Create new parts for the two new meshes.
            let positive_part = Part::new(clip_result.positive_mesh, parameters);
            let negative_part = Part::new(clip_result.negative_mesh, parameters);
            current_parts.push(positive_part);
            current_parts.push(negative_part);

            // Create the next state.
            let mut next_state =
                NodeState::from_parts(self.initial_part.clone(), current_costs, current_parts);
            next_state.current_value = Some((plane, self.worst_part_index as u32));
            next_state.update_score();
            next_state.current_cost = self.current_cost + next_state.current_score;
            next_state.current_round = self.current_round + 1;

            Some(next_state)
        } else {
            // If clipping failed, return a terminal state with max cost.
            let mut next_state = NodeState::from_parts(
                self.initial_part.clone(),
                self.current_costs.clone(),
                self.current_parts.clone(),
            );
            next_state.current_cost = f32::MAX;
            next_state.current_round = parameters.mcts_max_depth;

            Some(next_state)
        }
    }
}

/// A part of the mesh being considered for splitting.
#[derive(Clone)]
pub struct Part {
    pub current_mesh: IndexedMesh,
    pub current_aabb: Aabb,
    pub next_candidate: usize,
    pub candidate_planes: Vec<Plane>,
}

impl Part {
    #[inline]
    pub fn new(current_mesh: IndexedMesh, parameters: &CoacdParaneters) -> Self {
        // TODO: Should this be computed elsewhere?
        let current_aabb = current_mesh.compute_aabb();

        // Compute candidate planes.
        let mut candidate_planes = Vec::new();
        compute_axis_aligned_planes(current_aabb, parameters.mcts_nodes, &mut candidate_planes);

        Self {
            current_mesh,
            current_aabb,
            next_candidate: 0,
            candidate_planes,
        }
    }

    // TODO: This seems like it could be replaced with an iterator?
    /// Returns the next available move ([`Plane`]), and increments the internal counter.
    ///
    /// # Panics
    ///
    /// Panics if there are no more candidate planes available.
    #[inline]
    pub fn take_next_move(&mut self) -> Plane {
        let plane = self.candidate_planes[self.next_candidate];
        self.next_candidate += 1;
        plane
    }
}

/// Clips the given mesh by the sequence of planes in `best_path`,
/// and computes the average cost of the resulting parts.
///
/// Returns `None` if clipping fails at any point.
fn clip_by_path(
    mesh: &IndexedMesh,
    first_plane: &Plane,
    best_path: &[Plane],
    parameters: &CoacdParaneters,
) -> Option<f32> {
    // TODO: Store the scores and parts together
    let mut scores = Vec::with_capacity(best_path.len() + 1);
    let mut parts = Vec::with_capacity(best_path.len() + 1);

    let mut max_cost: f32;

    // Clip the initial mesh with the first plane.
    // TODO: Can't we just have this in the loop below?
    let clip_result = crate::clip::clip(mesh, first_plane)?;

    // TODO: Can we avoid converting to a mesh here?
    let positive_hull = clip_result
        .positive_mesh
        .compute_convex_hull()
        .ok()?
        .to_mesh();
    let negative_hull = clip_result
        .negative_mesh
        .compute_convex_hull()
        .ok()?
        .to_mesh();

    let positive_cost =
        parameters.rv_k * cost::compute_rv(&clip_result.positive_mesh, &positive_hull);
    let negative_cost =
        parameters.rv_k * cost::compute_rv(&clip_result.negative_mesh, &negative_hull);

    scores.push(positive_cost);
    scores.push(negative_cost);

    parts.push(clip_result.positive_mesh);
    parts.push(clip_result.negative_mesh);

    let (mut final_cost, mut worst_index) = if positive_cost > negative_cost {
        (positive_cost, 0)
    } else {
        (negative_cost, 1)
    };

    // Iteratively clip the worst part with the remaining planes in the best path.
    for plane in best_path.iter().skip(1).rev() {
        let clip_result = crate::clip::clip(&parts[worst_index], plane)?;

        // TODO: Can we avoid converting to a mesh here?
        let positive_hull = clip_result
            .positive_mesh
            .compute_convex_hull()
            .ok()?
            .to_mesh();
        let negative_hull = clip_result
            .negative_mesh
            .compute_convex_hull()
            .ok()?
            .to_mesh();

        let positive_cost =
            parameters.rv_k * cost::compute_rv(&clip_result.positive_mesh, &positive_hull);
        let negative_cost =
            parameters.rv_k * cost::compute_rv(&clip_result.negative_mesh, &negative_hull);

        scores.remove(worst_index);
        parts.remove(worst_index);

        scores.push(positive_cost);
        scores.push(negative_cost);

        parts.push(clip_result.positive_mesh);
        parts.push(clip_result.negative_mesh);

        // Reset `max_cost` and `worst_index` to find the new worst part.
        max_cost = scores[0];
        worst_index = 0;

        // Find the new worst part.
        for (j, &score) in scores.iter().enumerate().skip(1) {
            // TODO: Should this use `max_cost` or `final_cost`?
            if score > final_cost {
                max_cost = score;
                worst_index = j;
            }
        }

        final_cost += max_cost;
    }

    // Average the final cost over the number of parts.
    final_cost /= best_path.len() as f32;

    Some(final_cost)
}

fn ternary_mcts(
    mesh: &IndexedMesh,
    best_plane: &mut Plane,
    best_path: &[Plane],
    best_cost: f32,
    epsilon: f32,
    parameters: &CoacdParaneters,
) -> bool {
    if best_path.is_empty() {
        return false;
    }

    // TODO: Store the AABB or pass it as parameter.
    let aabb = mesh.compute_aabb();

    let interval = (aabb.diagonal() / (parameters.mcts_nodes + 1) as f32).max(Vec3A::splat(0.01));
    let min_interval = 0.01;
    let threshold = 10;

    // x-axis
    if (best_plane.normal().x - 1.0).abs() < 1e-4 {
        let mut left = (aabb.min.x + min_interval).max(-best_plane.d() - interval.x);
        let mut right = (aabb.max.x - min_interval).min(-best_plane.d() + interval.x);

        if left >= right {
            return false;
        }

        let mut i = 0;

        // Store the best result found during the search.
        let mut result = 0.0;

        // Perform ternary search to find the optimal plane position.
        while left + epsilon < right && i < threshold {
            let third = (right - left) / 3.0;
            let mid1 = left + third;
            let mid2 = right - third;

            let plane1 = Plane::from_coefficients(1.0, 0.0, 0.0, -mid1);
            let plane2 = Plane::from_coefficients(1.0, 0.0, 0.0, -mid2);

            let cost1 = clip_by_path(mesh, &plane1, best_path, parameters).unwrap_or(f32::MAX);
            let cost2 = clip_by_path(mesh, &plane2, best_path, parameters).unwrap_or(f32::MAX);

            if cost1 < cost2 {
                right = mid2;
                result = mid1;
            } else {
                left = mid1;
                result = mid2;
            }

            i += 1;
        }

        // Evaluate the best plane found during the search.
        let final_plane = Plane::from_coefficients(1.0, 0.0, 0.0, -result);
        let final_cost =
            clip_by_path(mesh, &final_plane, best_path, parameters).unwrap_or(f32::MAX);

        // Update the best plane if a better cost is found.
        if final_cost < best_cost {
            *best_plane = final_plane;
        }
    }

    // y-axis
    if (best_plane.normal().y - 1.0).abs() < 1e-4 {
        let mut left = (aabb.min.y + min_interval).max(-best_plane.d() - interval.y);
        let mut right = (aabb.max.y - min_interval).min(-best_plane.d() + interval.y);

        if left >= right {
            return false;
        }

        let mut i = 0;

        // Store the best result found during the search.
        let mut result = 0.0;

        // Perform ternary search to find the optimal plane position.
        while left + epsilon < right && i < threshold {
            let third = (right - left) / 3.0;
            let mid1 = left + third;
            let mid2 = right - third;

            let plane1 = Plane::from_coefficients(0.0, 1.0, 0.0, -mid1);
            let plane2 = Plane::from_coefficients(0.0, 1.0, 0.0, -mid2);

            let cost1 = clip_by_path(mesh, &plane1, best_path, parameters).unwrap_or(f32::MAX);
            let cost2 = clip_by_path(mesh, &plane2, best_path, parameters).unwrap_or(f32::MAX);

            if cost1 < cost2 {
                right = mid2;
                result = mid1;
            } else {
                left = mid1;
                result = mid2;
            }

            i += 1;
        }

        // Evaluate the best plane found during the search.
        let final_plane = Plane::from_coefficients(0.0, 1.0, 0.0, -result);
        let final_cost =
            clip_by_path(mesh, &final_plane, best_path, parameters).unwrap_or(f32::MAX);

        // Update the best plane if a better cost is found.
        if final_cost < best_cost {
            *best_plane = final_plane;
        }
    }

    // z-axis
    if (best_plane.normal().z - 1.0).abs() < 1e-4 {
        let mut left = (aabb.min.z + min_interval).max(-best_plane.d() - interval.z);
        let mut right = (aabb.max.z - min_interval).min(-best_plane.d() + interval.z);

        if left >= right {
            return false;
        }

        let mut i = 0;

        // Store the best result found during the search.
        let mut result = 0.0;

        // Perform ternary search to find the optimal plane position.
        while left + epsilon < right && i < threshold {
            let third = (right - left) / 3.0;
            let mid1 = left + third;
            let mid2 = right - third;

            let plane1 = Plane::from_coefficients(0.0, 0.0, 1.0, -mid1);
            let plane2 = Plane::from_coefficients(0.0, 0.0, 1.0, -mid2);

            let cost1 = clip_by_path(mesh, &plane1, best_path, parameters).unwrap_or(f32::MAX);
            let cost2 = clip_by_path(mesh, &plane2, best_path, parameters).unwrap_or(f32::MAX);

            if cost1 < cost2 {
                right = mid2;
                result = mid1;
            } else {
                left = mid1;
                result = mid2;
            }

            i += 1;
        }

        // Evaluate the best plane found during the search.
        let final_plane = Plane::from_coefficients(0.0, 0.0, 1.0, -result);
        let final_cost =
            clip_by_path(mesh, &final_plane, best_path, parameters).unwrap_or(f32::MAX);

        // Update the best plane if a better cost is found.
        if final_cost < best_cost {
            *best_plane = final_plane;
        }
    }

    true
}

/// Computes axis-aligned candidate planes within the given AABB,
/// and appends them to the provided `planes` vector.
///
/// The number of planes along each axis is determined by `num_nodes`.
/// The planes are spaced evenly within the AABB, avoiding planes too close to the edges.
#[inline]
fn compute_axis_aligned_planes(aabb: Aabb, num_nodes: u32, planes: &mut Vec<Plane>) {
    let extents = aabb.diagonal();
    let interval = (extents / (num_nodes as f32 + 1.0)).max(Vec3A::splat(0.01));
    let eps = 1e-6;
    let range_offset = interval.max(Vec3A::splat(0.015));

    // x-axis planes
    let mut i = aabb.min.x + range_offset.x;
    while i <= aabb.max.x - range_offset.x + eps {
        planes.push(Plane::from_coefficients(1.0, 0.0, 0.0, -i));
        i += interval.x;
    }

    // y-axis planes
    let mut j = aabb.min.y + range_offset.y;
    while j <= aabb.max.y - range_offset.y + eps {
        planes.push(Plane::from_coefficients(0.0, 1.0, 0.0, -j));
        j += interval.y;
    }

    // z-axis planes
    let mut k = aabb.min.z + range_offset.z;
    while k <= aabb.max.z - range_offset.z + eps {
        planes.push(Plane::from_coefficients(0.0, 0.0, 1.0, -k));
        k += interval.z;
    }
}

/// Finds the best splitting plane from a list of candidate planes that minimizes the
/// total `r_v` of the resulting two parts after clipping.
///
/// See [`cost`] for details on the `r_v` metric.
// TODO: The mixing of `r_v` and `rv` is annoying
fn find_best_rv_plane(
    mesh: &IndexedMesh,
    planes: &[Plane],
    parameters: &CoacdParaneters,
) -> Option<Plane> {
    if planes.is_empty() {
        return None;
    }

    let mut best_plane = None;
    let mut best_rv = f32::MAX;

    for &plane in planes {
        let Some(clip_result) = crate::clip::clip(mesh, &plane) else {
            best_rv = f32::MAX;
            continue;
        };

        if clip_result.positive_mesh.vertices.is_empty()
            || clip_result.negative_mesh.vertices.is_empty()
        {
            continue;
        }

        if let (Ok(positive_hull), Ok(negative_hull)) = (
            clip_result.positive_mesh.compute_convex_hull(),
            clip_result.negative_mesh.compute_convex_hull(),
        ) {
            // TODO: Can we avoid converting to a mesh here?
            let positive_hull = positive_hull.to_mesh();
            let negative_hull = negative_hull.to_mesh();

            // TODO: Technically we don't need to multiply by `rv_k` here since we're just comparing costs.
            let rv = parameters.rv_k
                * cost::compute_total_rv(
                    &clip_result.positive_mesh,
                    &positive_hull,
                    &clip_result.negative_mesh,
                    &negative_hull,
                );

            if rv < best_rv {
                best_rv = rv;
                best_plane = Some(plane);
            }
        }
    }

    best_plane
}
