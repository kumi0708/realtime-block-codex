import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/realtime-block-codex/",
  plugins: [react()],
  build: {
    rollupOptions: {
      external: ["react", "react-dom/client", "matter-js", "lucide-react"],
      output: {
        paths: {
          react: "https://esm.sh/react@19.2.1",
          "react-dom/client": "https://esm.sh/react-dom@19.2.1/client",
          "matter-js": "https://esm.sh/matter-js@0.20.0",
          "lucide-react": "https://esm.sh/lucide-react@0.562.0",
        },
      },
    },
  },
});
