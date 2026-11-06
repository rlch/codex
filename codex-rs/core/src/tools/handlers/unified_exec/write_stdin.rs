use crate::function_tool::FunctionCallError;
use codex_features::Feature;

use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::resolve_tool_environment;
use crate::tools::hook_names::HookToolName;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::PostToolUsePayload;
use crate::tools::registry::PreToolUsePayload;
use crate::tools::registry::ToolExecutor;
use crate::tools::sandboxing::ToolError;
use crate::unified_exec::UnifiedExecContext;
use crate::unified_exec::UnifiedExecError;
use crate::unified_exec::WriteStdinInteractionEvent;
use crate::unified_exec::WriteStdinRequest;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use serde::Deserialize;

use super::super::shell_spec::create_write_stdin_tool;
use super::post_unified_exec_tool_use_payload;

#[derive(Debug, Deserialize)]
struct WriteStdinArgs {
    // The model is trained on `session_id`.
    session_id: i32,
    #[serde(default)]
    chars: String,
    #[serde(default = "super::default_write_stdin_yield_time_ms")]
    yield_time_ms: u64,
    #[serde(default)]
    max_output_tokens: Option<usize>,
}

pub struct WriteStdinHandler;

impl ToolExecutor<ToolInvocation> for WriteStdinHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain("write_stdin")
    }

    fn spec(&self) -> ToolSpec {
        create_write_stdin_tool()
    }

    fn supports_parallel_tool_calls(&self) -> bool {
        true
    }

    fn handle<'a>(&'a self, invocation: ToolInvocation) -> codex_tools::ToolExecutorFuture<'a>
    where
        ToolInvocation: 'a,
    {
        Box::pin(self.handle_call(invocation))
    }
}

impl WriteStdinHandler {
    async fn handle_call(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ToolInvocation {
            session,
            turn,
            step_context,
            cancellation_token,
            call_id,
            payload,
            ..
        } = invocation;

        let arguments = match payload {
            ToolPayload::Function { arguments } => arguments,
            _ => {
                return Err(FunctionCallError::RespondToModel(
                    "write_stdin handler received unsupported payload".to_string(),
                ));
            }
        };

        let args: WriteStdinArgs = parse_arguments(&arguments)?;
        if turn
            .config
            .features
            .get()
            .enabled(Feature::StableEnvironmentTools)
        {
            resolve_tool_environment(
                &step_context,
                /*environment_id*/ None,
                "unified exec is unavailable in this session",
            )?;
        }
        let context =
            UnifiedExecContext::new(session.clone(), step_context, cancellation_token, call_id);
        let response = session
            .services
            .unified_exec_manager
            .write_stdin(
                &context,
                WriteStdinRequest {
                    process_id: args.session_id,
                    input: &args.chars,
                    yield_time_ms: args.yield_time_ms,
                    max_output_tokens: args.max_output_tokens,
                    truncation_policy: context
                        .step_context
                        .settings
                        .model_info
                        .truncation_policy
                        .into(),
                    interaction_event: Some(WriteStdinInteractionEvent {
                        session: &session,
                        turn: &turn,
                    }),
                },
            )
            .await
            .map_err(|err| {
                let message = match err {
                    UnifiedExecError::StdinApproval(ToolError::Rejected(reason)) => {
                        format!("write_stdin rejected: {reason}")
                    }
                    UnifiedExecError::StdinApproval(ToolError::Codex(err)) => {
                        format!("write_stdin approval failed: {err}")
                    }
                    err => format!("write_stdin failed: {err}"),
                };
                FunctionCallError::RespondToModel(message)
            })?;

        Ok(boxed_tool_output(response))
    }
}

impl CoreToolRuntime for WriteStdinHandler {
    fn matches_kind(&self, payload: &ToolPayload) -> bool {
        matches!(payload, ToolPayload::Function { .. })
    }

    fn pre_tool_use_payload(&self, invocation: &ToolInvocation) -> Option<PreToolUsePayload> {
        // abyme: what is typed into a terminal left open is a command of its own (a
        // shell started by `exec_command` runs any line written to it), so a non-empty
        // write asks PreToolUse as Bash, its input the command. Empty writes are
        // background polls and ask nothing.
        let ToolPayload::Function { arguments } = &invocation.payload else {
            return None;
        };
        let args: WriteStdinArgs = parse_arguments(arguments).ok()?;
        if args.chars.is_empty() {
            return None;
        }
        Some(PreToolUsePayload {
            tool_name: HookToolName::bash(),
            tool_input: serde_json::json!({ "command": args.chars }),
        })
    }

    fn with_updated_hook_input(
        &self,
        mut invocation: ToolInvocation,
        updated_input: serde_json::Value,
    ) -> Result<ToolInvocation, FunctionCallError> {
        let ToolPayload::Function { arguments } = invocation.payload else {
            return Err(FunctionCallError::RespondToModel(
                "hook input rewrite received unsupported write_stdin payload".to_string(),
            ));
        };
        invocation.payload = ToolPayload::Function {
            arguments: crate::tools::handlers::rewrite_function_string_argument(
                &arguments,
                "write_stdin",
                "chars",
                crate::tools::handlers::updated_hook_command(&updated_input)?,
            )?,
        };
        Ok(invocation)
    }

    fn post_tool_use_payload(
        &self,
        invocation: &ToolInvocation,
        result: &dyn crate::tools::context::ToolOutput,
    ) -> Option<PostToolUsePayload> {
        // A `write_stdin` poll can observe final completion for the original
        // `exec_command`; emit that command's matching Bash PostToolUse.
        post_unified_exec_tool_use_payload(invocation, result)
    }
}
