/** @type {import('tailwindcss').Config} */

module.exports = {
  content: ["./src/**/*.{html,js,jsx,civet}"],
  plugins: [
    require("@tailwindcss/typography"),
    require("daisyui")
  ],
};
