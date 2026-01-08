/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/renderer/**/*.{js,ts,jsx,tsx,html}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#081710',
          primary: '#123524',
          light: '#FAFAFA',
          accent: '#D64933',
        },
      },
      fontFamily: {
        manrope: ['Manrope', 'system-ui', 'sans-serif'],
      },
      backdropBlur: {
        '3xl': '64px',
      },
      animation: {
        'rainbow': 'rainbow 2s linear infinite',
      },
      keyframes: {
        rainbow: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
      },
    },
  },
  plugins: [],
}
