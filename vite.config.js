import { defineConfig } from "vite";

// Relative base so the build works both at / and under a GitHub Pages subpath (/ricky-pet/).
export default defineConfig({
  base: "./",
});
