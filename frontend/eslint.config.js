import js from "@eslint/js";

export default [
  js.configs.recommended,
  {
    files: ["**/*.js"],
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: {
        document: "readonly",
        fetch: "readonly",
        FormData: "readonly",
        Set: "readonly",
        Number: "readonly",
        Error: "readonly",
        JSON: "readonly",
      },
    },
  },
];
