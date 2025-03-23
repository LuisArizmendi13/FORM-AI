import { z } from "zod";
import { defineDAINService, ToolConfig } from "@dainprotocol/service-sdk";
import { DainResponse, CardUIBuilder } from "@dainprotocol/utils";
import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs";
import path from "path";
import axios from "axios"; 
import os from 'os'; 

const execAsync = promisify(exec);

const getPhysicalHealthDataConfig: ToolConfig = {
  id: "getPhysicalHealthData",
  name: "Get Physical Health Data",
  description: "Analyzes an MP4 video and returns the type of exercise",
  input: z.object({
    videoFile: z.string().describe("DAIN storage URL of the uploaded MP4 file"),
  }),
  output: z.object({
    typeOfExercise: z.string().describe("Detected type of exercise in the video"),
  }),
  handler: async ({ videoFile }, agentInfo) => {
    const tempFilePath = path.join(os.tmpdir(), "temp_video.mp4");

    try {
      // ✅ Download the video using axios
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

      // ✅ Call the Python integration script
      const scriptPath = path.join(__dirname, "integration.py");
      const { stdout, stderr } = await execAsync(`python3 "${scriptPath}" "${tempFilePath}"`);

      if (stderr) {
        console.warn("Python script stderr (non-blocking):", stderr);
      }

      const typeOfExercise = stdout.trim();

      const cardUI = new CardUIBuilder()
        .title("Exercise Detection Result")
        .content(`Detected Exercise: **${typeOfExercise}**`)
        .build();

      return new DainResponse({
        text: `The detected type of exercise in the video is: ${typeOfExercise}`,
        data: { typeOfExercise },
        ui: cardUI,
      });
    } catch (error) {
      console.error("❌ Error in getPhysicalHealthData:", error);
      return new DainResponse({
        text: "❌ Error processing the video.",
        data: { error: "An error occurred while analyzing the video." },
        ui: new CardUIBuilder()
          .title("Error")
          .content("An error occurred while processing the video.")
          .build(),
      });
    } finally {
      // 🧹 Clean up temp file
      try {
        if (fs.existsSync(tempFilePath)) {
          fs.unlinkSync(tempFilePath);
        }
      } catch (cleanupErr) {
        console.warn("Temp cleanup failed:", cleanupErr);
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
    tags:['excercise', 'health', 'fitness'], 
    logo: "https://cdn-icons-png.flaticon.com/512/252/252035.png"
  },
  identity: {
    apiKey: process.env.DAIN_API_KEY,
  },
  tools: [getPhysicalHealthDataConfig],
});

dainService.startNode().then(({ address }) => {
  console.log("🏃 FORMAI Service running on port:", address().port);
});
