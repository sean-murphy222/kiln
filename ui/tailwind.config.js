/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Kiln forge palette — industrial precision with warmth
        'kiln': {
          '950': '#080A0E',
          '900': '#0C0E12',
          '800': '#161921',
          '700': '#1E222D',
          '600': '#2A3040',
          '500': '#4A5468',
          '400': '#8892A8',
          '300': '#B8C0D0',
          '200': '#E2E6EF',
          '100': '#F4F5F8',
        },

        // Primary accent — warm ember
        'ember': {
          DEFAULT: '#E8734A',
          'glow': '#F59E6C',
          'dim': '#C4563A',
          'faint': 'rgba(232, 115, 74, 0.08)',
          'subtle': 'rgba(232, 115, 74, 0.15)',
        },

        // Tool identity colors
        'quarry': {
          DEFAULT: '#7C92A8',
          'light': '#9DB1C4',
          'faint': 'rgba(124, 146, 168, 0.10)',
        },
        'forge-heat': {
          DEFAULT: '#D4915C',
          'light': '#E0AD80',
          'faint': 'rgba(212, 145, 92, 0.10)',
        },
        'foundry-cast': {
          DEFAULT: '#6BA089',
          'light': '#8DB8A5',
          'faint': 'rgba(107, 160, 137, 0.10)',
        },
        'hearth-glow': {
          DEFAULT: '#D4A058',
          'light': '#E0BA80',
          'faint': 'rgba(212, 160, 88, 0.10)',
        },

        // Semantic colors
        'success': '#5CB87A',
        'warning': '#E0A84C',
        'error': '#D45B5B',
        'info': '#5B9BD4',

      },
      fontFamily: {
        'display': ['"DM Sans"', 'system-ui', 'sans-serif'],
        'body': ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        'mono': ['"IBM Plex Mono"', 'monospace'],
        'sans': ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.875rem' }],
      },
      boxShadow: {
        'kiln-sm': '0 1px 2px rgba(0, 0, 0, 0.3)',
        'kiln': '0 2px 8px rgba(0, 0, 0, 0.3)',
        'kiln-lg': '0 4px 16px rgba(0, 0, 0, 0.4)',
        'kiln-glow': '0 0 20px rgba(232, 115, 74, 0.15)',
        'card-edge': 'inset 0 1px 0 rgba(255, 255, 255, 0.04)',
      },
      borderRadius: {
        'kiln': '6px',
      },
      animation: {
        'fade-in': 'fade-in 0.2s ease-out',
        'slide-up': 'slide-up 0.2s ease-out',
        'slide-down': 'slide-down 0.2s ease-out',
        'slide-right': 'slide-right 0.2s ease-out',
        'pulse-soft': 'pulse-soft 2s ease-in-out infinite',
      },
      keyframes: {
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'slide-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-down': {
          '0%': { opacity: '0', transform: 'translateY(-8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-right': {
          '0%': { opacity: '0', transform: 'translateX(-8px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        'pulse-soft': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
      },
      spacing: {
        'nav-rail': '64px',
      },
    },
  },
  plugins: [],
}
