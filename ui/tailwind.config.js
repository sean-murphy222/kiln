/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 8-bit retro palette
        'chonk': {
          // Primary colors
          'black': '#1a1c2c',
          'dark': '#5d275d',
          'purple': '#b13e53',
          'red': '#ef7d57',
          'orange': '#ffcd75',
          'yellow': '#a7f070',
          'green': '#38b764',
          'teal': '#257179',
          'blue': '#29366f',
          'navy': '#3b5dc9',
          'sky': '#41a6f6',
          'cyan': '#73eff7',
          'white': '#f4f4f4',
          'light': '#94b0c2',
          'gray': '#566c86',
          'slate': '#333c57',
        },
        // Semantic colors
        'surface': {
          'bg': '#1a1c2c',
          'panel': '#333c57',
          'card': '#566c86',
          'hover': '#94b0c2',
        },
        'accent': {
          'primary': '#41a6f6',
          'secondary': '#b13e53',
          'success': '#38b764',
          'warning': '#ffcd75',
          'error': '#ef7d57',
        },
      },
      fontFamily: {
        'pixel': ['"Press Start 2P"', 'monospace'],
        'mono': ['JetBrains Mono', 'monospace'],
        'sans': ['Inter', 'sans-serif'],
      },
      boxShadow: {
        'pixel': '4px 4px 0px 0px rgba(0, 0, 0, 0.25)',
        'pixel-sm': '2px 2px 0px 0px rgba(0, 0, 0, 0.25)',
        'pixel-inset': 'inset 2px 2px 0px 0px rgba(0, 0, 0, 0.25)',
      },
      borderWidth: {
        'pixel': '3px',
      },
      animation: {
        'bounce-pixel': 'bounce-pixel 0.5s steps(2) infinite',
        'pulse-pixel': 'pulse-pixel 1s steps(2) infinite',
      },
      keyframes: {
        'bounce-pixel': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-4px)' },
        },
        'pulse-pixel': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.5' },
        },
      },
    },
  },
  plugins: [],
}
