import { z } from "zod";
import { defineDAINService, ToolConfig } from "@dainprotocol/service-sdk";
import { DainResponse, CardUIBuilder } from "@dainprotocol/utils";
import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs";
import path from "path";
import axios from "axios";
import os from "os";

const execAsync = promisify(exec);

const getPhysicalHealthDataConfig: ToolConfig = {
  id: "getPhysicalHealthData",
  name: "Get Physical Health Data",
  description: "Analyzes an MP4 video and returns results with images and recommendations",
  input: z.object({
    videoFile: z.string().describe("DAIN storage URL of the uploaded MP4 file"),
  }),
  output: z.object({
    typeOfExercise: z.string().describe("Detected type of exercise"),
    followUp: z.string().describe("Suggested improvements or advice"),
    previewUrl: z.string().describe("URL to skeleton comparison image"),
    graphUrl: z.string().describe("URL to error graph image"),
  }),
  handler: async ({ videoFile }) => {
    const tempFilePath = path.join(os.tmpdir(), "temp_video.mp4");

    try {
      const response = await axios({
        method: "get",
        url: videoFile,
        responseType: "stream",
      });

      const writer = fs.createWriteStream(tempFilePath);
      response.data.pipe(writer);

      await new Promise<void>((resolve, reject) => {
        writer.on("finish", resolve);
        writer.on("error", reject);
      });

      const scriptPath = path.join(__dirname, "integration.py");
      const { stdout, stderr } = await execAsync(`python3 "${scriptPath}" "${tempFilePath}"`, {
        cwd: path.resolve(__dirname, "../../../"), // 🧭 run from root
      });

      if (stderr) console.warn("⚠️ Python stderr:", stderr);

      const lines = stdout.trim().split("\n");
      const [previewUrl, graphUrl, followUp, typeOfExercise] = lines.slice(-4);

      const cardUI = new CardUIBuilder()
        .title("Exercise Analysis")
        .content(
          `**Detected Exercise:** ${typeOfExercise}\n\n` +
          `**Improvement Advice:**\n${followUp}\n\n` +
          `**Skeleton Comparison:** [View](${previewUrl})\n\n` +
          `**Error Graph:** [View](${graphUrl})`
        )
        .build();

      return new DainResponse({
        text: `Detected ${typeOfExercise}. Advice and visuals included.`,
        data: {
          typeOfExercise,
          followUp,
          previewUrl,
          graphUrl,
        },
        ui: cardUI,
      });
    } catch (error) {
      console.error("❌ Error in handler:", error);
      return new DainResponse({
        text: "An error occurred while analyzing the video.",
        data: {
          typeOfExercise: "unknown",
          followUp: "No suggestions due to an error.",
          previewUrl: "",
          graphUrl: "",
        },
        ui: new CardUIBuilder()
          .title("Error")
          .content("There was a problem processing the video.")
          .build(),
      });
    } finally {
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
    }
  },
};

const dainService = defineDAINService({
  metadata: {
    title: "FORMAI",
    description: "A service that analyzes physical health data from videos",
    version: "1.0.0",
    author: "FORM_AI",
    tags: ["exercise", "health", "fitness"],
    logo: "https://cdn-icons-png.flaticon.com/512/252/252035.png",
  },
  identity: {
    apiKey: process.env.DAIN_API_KEY,
  },
  tools: [getPhysicalHealthDataConfig],
});

dainService.startNode().then(({ address }) => {
  console.log("🏃 FORMAI Service running on port:", address().port);
});
