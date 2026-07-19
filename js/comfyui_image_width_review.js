import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "NanGuangAI.ImageWidthReview",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "Comfyui_Image_Width_Review") {
            return;
        }
        nodeData.display_name = "南光AI宽限图像";
        // 保持系统默认配色，不覆盖
    }
});

console.log("[南光AI] 宽限图像节点已加载（完整版）");