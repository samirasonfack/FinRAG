import express from "express";
import dotenv from "dotenv";
import Anthropic from "@anthropic-ai/sdk";
import fs from "fs-extra";
import cors from "cors";

dotenv.config();

const app = express();
app.use(express.json());
app.use(cors()); // autorise le frontend

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

app.post("/api/claude", async (req, res) => {
  const { prompt } = req.body;

  const response = await client.messages.create({
    model: "claude-3-5-sonnet-20241022",
    messages: [{ role: "user", content: prompt }],
    max_tokens: 1000,
  });

  // Optionnel : sauvegarde automatique
  await fs.outputFile(`./output/GeneratedCode.js`, response.content);

  res.json({ content: response.content });
});

app.listen(3001, () => console.log("Backend Claude running on http://localhost:3001"));
