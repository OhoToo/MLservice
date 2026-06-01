import { accessSync, constants } from "node:fs";

const requiredFiles = ["index.html", "script.js", "style.css"];

for (const file of requiredFiles) {
  accessSync(file, constants.R_OK);
}

console.log("Static frontend files are present.");
