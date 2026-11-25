"use strict";

function createDensityView(host, paletteColor) {
  const get = id => document.getElementById(id);
  const status = get("volume-status");
  const sliceToggle = get("volume-slice");
  const layerReadout = get("volume-layer");
  const controls = [sliceToggle, get("volume-zoom-out"), get("volume-zoom-in"), get("volume-reset")];
  const T = window.THREE;
  let renderer;
  try {
    if (!T) throw new Error("Three.js is unavailable");
    renderer = new T.WebGLRenderer({antialias: true});
  } catch {
    status.textContent = "当前浏览器无法显示三维视图，仍可查看水平切片。";
    return {setModel() {}, setLayer() {}};
  }

  host.append(renderer.domElement);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene = new T.Scene();
  scene.background = new T.Color("#f3f7f9");
  const camera = new T.OrthographicCamera(-1, 1, 1, -1, .1, 300);
  const material = new T.MeshBasicMaterial({vertexColors: true});
  const tint = new T.Color();
  let model, slice, bounds, layer = 0, labels = [], frame = 0, drag;
  let azimuth = -.66, elevation = .57, zoom = 1;

  // Grid order is [X, Y, depth]; world up is opposite to depth.
  function point(x, y, depth) {
    return [x - bounds[0] / 2, bounds[2] / 2 - depth, y - bounds[1] / 2];
  }

  function surface(density, n, limit) {
    const positions = [], colors = [], indices = [];
    const stride = [n * n, n, 1], step = bounds.map(length => length / n);
    const range = [-limit, limit], rgb = [0, 0, 0], colorCache = new Map();
    // Match the legend on top faces and shade only the brightness of the other faces.
    const faceShade = [[.79, .73], [.66, .83], [1, .57]];
    for (let x = 0; x < n; x++) for (let y = 0; y < n; y++) for (let z = 0; z < n; z++) {
      const cell = [x, y, z], index = x * stride[0] + y * stride[1] + z;
      const value = density[index];
      if (!value) continue;
      let color = colorCache.get(value);
      if (!color) {
        paletteColor(value, range, rgb);
        color = tint.setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, T.SRGBColorSpace).toArray();
        colorCache.set(value, color);
      }
      for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) {
        const adjacent = cell[axis] + sign;
        if (adjacent >= 0 && adjacent < n && density[index + sign * stride[axis]]) continue;
        const a = (axis + 1) % 3, b = (axis + 2) % 3, start = positions.length / 3;
        const shade = faceShade[axis][sign > 0 ? 1 : 0];
        const corner = cell.map((i, d) => i * step[d]);
        if (sign > 0) corner[axis] += step[axis];
        for (const [u, v] of [[0, 0], [1, 0], [1, 1], [0, 1]]) {
          const p = corner.slice();
          p[a] += u * step[a];
          p[b] += v * step[b];
          positions.push(...point(...p));
          colors.push(color[0] * shade, color[1] * shade, color[2] * shade);
        }
        if (sign > 0) indices.push(start, start+1, start+2, start, start+2, start+3);
        else indices.push(start, start+2, start+1, start, start+3, start+2);
      }
    }
    const geometry = new T.BufferGeometry();
    geometry.setAttribute("position", new T.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("color", new T.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    return new T.Mesh(geometry, material);
  }

  function lineGeometry(segments) {
    const geometry = new T.BufferGeometry();
    geometry.setAttribute("position", new T.Float32BufferAttribute(segments.flat(2), 3));
    return geometry;
  }

  function frameModel() {
    const [x, y, depth] = bounds;
    const solid = new T.BoxGeometry(x, depth, y);
    const edges = new T.EdgesGeometry(solid);
    solid.dispose();
    model.add(new T.LineSegments(edges, new T.LineBasicMaterial({color: 0x9cafb7})));
    const grid = [];
    for (let i = 1; i < 5; i++) {
      grid.push([point(x*i/5, 0, depth), point(x*i/5, y, depth)]);
      grid.push([point(0, y*i/5, depth), point(x, y*i/5, depth)]);
    }
    model.add(new T.LineSegments(lineGeometry(grid), new T.LineBasicMaterial({color: 0xd0dce0})));

    const surfaceY = depth / 2;
    slice = new T.Group();
    const plane = new T.Mesh(new T.PlaneGeometry(x, y), new T.MeshBasicMaterial({
      color: 0x169989, opacity: .17, transparent: true, depthWrite: false, side: T.DoubleSide
    }));
    plane.rotation.x = -Math.PI / 2;
    slice.add(plane);
    const corners = [point(0, 0, 0), point(x, 0, 0), point(x, y, 0), point(0, y, 0)];
    corners.forEach(p => { p[1] -= surfaceY; });
    slice.add(new T.LineLoop(lineGeometry([corners]), new T.LineBasicMaterial({color: 0x008a78})));
    model.add(slice);
    for (const [label, x0, y0, d0] of [
      ["X " + x + " km", x/2, -.9, depth], ["Y " + y + " km", -.9, y/2, depth],
      ["深度 " + depth + " km", -.35, y+.35, depth/2]
    ]) {
      const canvas = document.createElement("canvas");
      canvas.width = 208;
      canvas.height = 54;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#344c55";
      ctx.font = '600 28px "Segoe UI", "Microsoft YaHei", sans-serif';
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(label, 104, 28);
      const map = new T.CanvasTexture(canvas);
      map.colorSpace = T.SRGBColorSpace;
      const tag = new T.Sprite(new T.SpriteMaterial({map, depthTest: false}));
      tag.position.set(...point(x0, y0, d0));
      tag.renderOrder = 5;
      labels.push(tag);
      model.add(tag);
    }
  }

  function draw() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      const {clientWidth: width, clientHeight: height} = host;
      if (!(width && height)) return;
      const aspect = width / height;
      const radius = bounds ? Math.hypot(...bounds) / 2 : 15;
      const padding = 1.20 + .23 * Math.max(0, 1.55 - aspect);
      const half = radius * padding / Math.min(1.55, aspect) / zoom;
      camera.left = -half * aspect;
      camera.right = half * aspect;
      camera.top = half;
      camera.bottom = -half;
      camera.position.set(
        60 * Math.cos(elevation) * Math.sin(azimuth),
        60 * Math.sin(elevation),
        -60 * Math.cos(elevation) * Math.cos(azimuth)
      );
      camera.lookAt(0, 0, 0);
      camera.updateProjectionMatrix();
      const unit = 2 * half / height;
      for (const tag of labels) tag.scale.set(104 * unit, 27 * unit, 1);
      const size = renderer.getSize(new T.Vector2());
      if (size.x !== width || size.y !== height) renderer.setSize(width, height, false);
      renderer.render(scene, camera);
    });
  }

  function setLayer(index, n) {
    if (Number.isFinite(index)) layer = index;
    if (!slice) return;
    const km = (layer + .5) * bounds[2] / n;
    slice.position.y = bounds[2] / 2 - km;
    slice.visible = sliceToggle.checked;
    layerReadout.hidden = !slice.visible;
    layerReadout.textContent = "切片 " + km.toFixed(2) + " km";
    draw();
  }

  let gridSize = 0;
  function setModel(density, n, limits, limit) {
    if (model) {
      scene.remove(model);
      model.traverse(obj => {
        obj.geometry?.dispose();
        if (obj.material && obj.material !== material) {
          obj.material.map?.dispose();
          obj.material.dispose();
        }
      });
      model = slice = undefined;
      labels = [];
    }
    controls.forEach(control => { control.disabled = !density; });
    host.dataset.ready = String(Boolean(density));
    status.hidden = Boolean(density);
    layerReadout.hidden = true;
    if (density) {
      gridSize = n;
      bounds = limits.map(length => length / 1000);
      model = new T.Group();
      model.add(surface(density, n, limit));
      frameModel();
      scene.add(model);
      setLayer(layer, n);
    } else {
      status.textContent = "尚无密度模型";
    }
    draw();
  }

  function spin(dx, dy) {
    azimuth -= dx;
    elevation = Math.max(-1.38, Math.min(1.38, elevation + dy));
    draw();
  }
  function scale(factor) {
    zoom = Math.max(.62, Math.min(2.8, zoom * factor));
    draw();
  }
  host.addEventListener("pointerdown", event => {
    if (!model || drag || event.button !== 0) return;
    host.setPointerCapture(event.pointerId);
    drag = {id: event.pointerId, x: event.clientX, y: event.clientY};
    host.dataset.dragging = "true";
  });
  host.addEventListener("pointermove", event => {
    if (drag?.id !== event.pointerId) return;
    spin((event.clientX - drag.x) * .008, (event.clientY - drag.y) * .008);
    drag.x = event.clientX;
    drag.y = event.clientY;
  });
  host.addEventListener("lostpointercapture", event => {
    if (drag?.id !== event.pointerId) return;
    drag = undefined;
    delete host.dataset.dragging;
  });
  host.addEventListener("wheel", event => {
    if (!model || event.ctrlKey) return;
    event.preventDefault();
    scale(Math.exp(-Math.sign(event.deltaY) * .11));
  }, {passive: false});
  host.addEventListener("keydown", event => {
    if (!model) return;
    const directions = {ArrowLeft: [-.15, 0], ArrowRight: [.15, 0], ArrowUp: [0, -.15], ArrowDown: [0, .15]};
    if (event.key in directions) spin(...directions[event.key]);
    else if (event.key === "+" || event.key === "=") scale(1.16);
    else if (event.key === "-") scale(1/1.16);
    else return;
    event.preventDefault();
  });
  get("volume-reset").addEventListener("click", () => {
    azimuth = -.66;
    elevation = .57;
    zoom = 1;
    draw();
  });
  get("volume-zoom-in").addEventListener("click", () => scale(1.16));
  get("volume-zoom-out").addEventListener("click", () => scale(1/1.16));
  sliceToggle.addEventListener("change", () => setLayer(layer, gridSize));
  new ResizeObserver(draw).observe(host);
  return {setModel, setLayer};
}
