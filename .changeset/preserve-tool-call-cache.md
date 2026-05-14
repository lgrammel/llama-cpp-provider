---
"@lgrammel/llama-cpp-provider": patch
---

Preserve the exact generated tool-call JSON when reusing cached prompts so follow-up tool-result turns can benefit from prefix cache hits.
