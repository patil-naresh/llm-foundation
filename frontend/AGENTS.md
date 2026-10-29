<!-- LOVABLE:BEGIN -->
> [!IMPORTANT]
> This project is connected to [Lovable](https://lovable.dev). Avoid rewriting
> published git history — force pushing, or rebasing/amending/squashing commits
> that are already pushed — as it rewrites history on Lovable's side and the
> user will likely lose their project history.
>
> Commits you push to the connected branch sync back to Lovable and show up in
> the editor, so keep the branch in a working state.
<!-- LOVABLE:END -->

- Keep FastAPI in the sibling `../backend/` directory as a separately run Python service; the TanStack UI consumes it via HTTP so Python is not bundled into the UI host.
- Use an application factory and environment-only provider credentials; tests inject configuration without a real provider secret.
- Normalize provider streams to text and explicit error events; the UI surfaces SSE failures instead of silently accepting partial replies.
