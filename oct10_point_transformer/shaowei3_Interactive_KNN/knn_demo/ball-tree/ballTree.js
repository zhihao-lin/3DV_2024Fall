function Node(points, centroid, radius, parent = null) {
  this.points = points;  // Points within this node
  this.centroid = centroid;  // Center of the ball (circle)
  this.radius = radius;  // Radius of the ball (circle)
  this.parent = parent;
  this.left = null;
  this.right = null;
}

function BallTree() {
  this.root = null;  // Start with an empty tree

  // Insert a new point incrementally
  this.insert = function(point) {
      if (!this.root) {
          // If the tree is empty, create the root node
          this.root = new Node([point], point, 0);
      } else {
          // Insert the point into the correct location
          this._insertPoint(this.root, point);
      }
  };

  // Recursively insert the point into the tree
  this._insertPoint = function(node, point) {
      if (node.points.length === 1) {
          // Split the current leaf node
          const oldPoint = node.points[0];
          node.points = [oldPoint, point];  // Add the new point temporarily

          // Recalculate centroid and radius for the new node
          const centroid = calculateCentroid(node.points);
          const radius = calculateRadius(node.points, centroid);
          node.centroid = centroid;
          node.radius = radius;

          // Split the points into two new nodes
          const [leftPoints, rightPoints] = splitPoints(node.points, centroid);
          node.left = new Node(leftPoints, calculateCentroid(leftPoints), calculateRadius(leftPoints, calculateCentroid(leftPoints)), node);
          node.right = new Node(rightPoints, calculateCentroid(rightPoints), calculateRadius(rightPoints, calculateCentroid(rightPoints)), node);
          node.points = [];  // Clear points from the parent node
      } else {
          // Recursively insert into the left or right child
          const leftDist = distance(point, node.left.centroid);
          const rightDist = distance(point, node.right.centroid);

          if (leftDist < rightDist) {
              this._insertPoint(node.left, point);
          } else {
              this._insertPoint(node.right, point);
          }

          // Recalculate the centroid and radius for the parent node
          const points = this.collectPoints(node);
          node.centroid = calculateCentroid(points);
          node.radius = calculateRadius(points, node.centroid);
      }
  };

  // Collect all points in the tree
  this.collectPoints = function(node) {
      if (!node || !node.left && !node.right) return node.points;
      return this.collectPoints(node.left).concat(this.collectPoints(node.right));
  };

  // Convert the Ball Tree into a format compatible with D3 (with children)
  this.convertToD3Tree = function(node) {
      if (!node) return null;

      const d3Node = {
          centroid: node.centroid,
          children: []
      };

      if (node.left) d3Node.children.push(this.convertToD3Tree(node.left));
      if (node.right) d3Node.children.push(this.convertToD3Tree(node.right));

      if (d3Node.children.length === 0) delete d3Node.children;  // If no children, remove the property

      return d3Node;
  };

  // Method to visualize the tree (returns the root node in D3-compatible format)
  this.visualize = function() {
      return this.convertToD3Tree(this.root);
  };

  // Calculate the centroid of a set of points
  function calculateCentroid(points) {
      const numDimensions = points[0].length;
      const centroid = new Array(numDimensions).fill(0);

      points.forEach(point => {
          for (let i = 0; i < numDimensions; i++) {
              centroid[i] += point[i];
          }
      });

      return centroid.map(val => val / points.length);
  }

  // Calculate the radius of a node
  function calculateRadius(points, centroid) {
      return Math.max(...points.map(point => distance(point, centroid)));
  }

  // Split the points into two groups based on distance from the centroid
  function splitPoints(points, centroid) {
      points.sort((a, b) => distance(a, centroid) - distance(b, centroid));
      const medianIdx = Math.floor(points.length / 2);
      return [points.slice(0, medianIdx), points.slice(medianIdx)];
  }

  // Euclidean distance between two points
  function distance(point1, point2) {
      return Math.sqrt(point1.reduce((sum, val, i) => sum + (val - point2[i]) ** 2, 0));
  }
}
