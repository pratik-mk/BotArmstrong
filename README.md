# Rocket Landing

A static page that learns to land a rocket in the browser. The network starts with random weights. Physics, reward shaping, and Double DQN updates are the only things that change how it flies.

## Run locally

```bash
npm install
npm run dev
```

Open the URL Vite prints. Use 16× to practice, 1× to watch a flight, and Space to pause.

`npm run train-check` runs the same physics and network headless and checks that later flights score higher than early ones and that some of them land.

## Host it

```bash
npm run build
```

`dist/` is a static site. `base` is relative, so the same folder works on GitHub Pages and Vercel.

- **Vercel:** import the repo, framework preset Vite, output directory `dist`.
- **GitHub Pages:** build, then publish the contents of `dist/`. From a project site, the relative asset paths still resolve.

Training happens in the visitor's tab. Nothing is uploaded.
