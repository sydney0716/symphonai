+++
name = "memory-review"
description = "Review saved memories: find outdated, conflicting and duplicate ones, and promote lasting rules into the project's instructions."
when_to_use = "When the person asks to review, clean up or organise memories, or to move what the agent has learned into the instructions."
+++
# Memory review: tidy what SymphonAI remembers

1. **Read.** Read the saved memories in your context, both project and global, and `.symphonai/INSTRUCTIONS.md` if it exists.
2. **Sort each memory into one group:**
   - **Promote:** a lasting rule for this project that every agent should follow, and that belongs in the instructions.
   - **Merge:** two or more memories saying the same thing.
   - **Conflict:** memories that disagree, with each other or with the instructions.
   - **Outdated:** facts the code or the conversation shows are no longer true.
   - **Keep:** everything else.
3. **Propose before changing anything.** Show the groups as one list, each item with its memory id and the change you propose. Ask the person to approve, edit or reject with ask_user. For a conflict, ask which side is right.
4. **Apply only what was approved.**
   - Merge with `remember` and `replaces`, then `forget` the rest.
   - Outdated memories: `forget`.
   - Promotions: hand the new lines for `.symphonai/INSTRUCTIONS.md` to the implementer as a spec, then `forget` the promoted memories.
5. **Report** what changed, in one short list.
