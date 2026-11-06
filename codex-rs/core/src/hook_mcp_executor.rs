use std::sync::Arc;

use anyhow::bail;
use codex_hooks::HookMcpCall;
use codex_hooks::HookMcpExecutor;
use codex_mcp::McpRuntime;
use codex_protocol::ThreadId;
use futures::FutureExt;
use futures::future::BoxFuture;
use serde_json::Value;

pub(crate) struct CoreHookMcpExecutor {
    pub(crate) runtime: Arc<McpRuntime>,
    // Session-scoped MCP tools require the owning thread ID in request metadata.
    pub(crate) thread_id: ThreadId,
}

impl HookMcpExecutor for CoreHookMcpExecutor {
    fn execute(&self, call: HookMcpCall) -> BoxFuture<'_, anyhow::Result<String>> {
        async move {
            let mut metadata = call.metadata.unwrap_or_default();
            metadata.insert(
                "threadId".to_string(),
                Value::String(self.thread_id.to_string()),
            );

            let result = self
                .runtime
                .latest_call_tool(
                    &call.server,
                    &call.tool,
                    call.environment_id.as_deref(),
                    Some(Value::Object(call.input)),
                    Some(Value::Object(metadata)),
                    Some(call.timeout),
                    /*wait_for_server*/ false,
                )
                .await?;
            let text = result
                .content
                .iter()
                .filter_map(|content| {
                    (content.get("type").and_then(Value::as_str) == Some("text"))
                        .then(|| content.get("text").and_then(Value::as_str))
                        .flatten()
                })
                .collect::<Vec<_>>()
                .join("\n");
            if result.is_error == Some(true) {
                bail!("MCP tool returned an error: {text}");
            }

            Ok(text)
        }
        .boxed()
    }
}

/// An embedder's executor for the hook MCP calls to the servers it claims.
///
/// A host that runs Codex in its own process can answer some hooks itself, as `mcp_tool` hooks on
/// a server name only it knows. Calls to any other server go to Codex's own executor.
#[derive(Clone)]
pub struct EmbedderHookMcpExecutor {
    /// The server names this executor answers.
    pub servers: Vec<String>,
    pub executor: Arc<dyn HookMcpExecutor>,
}

/// Sends a call to a server the embedder claims to its executor, and every other to core's.
pub(crate) struct RoutingHookMcpExecutor {
    pub(crate) embedder: Option<EmbedderHookMcpExecutor>,
    pub(crate) core: CoreHookMcpExecutor,
}

impl HookMcpExecutor for RoutingHookMcpExecutor {
    fn execute(&self, call: HookMcpCall) -> BoxFuture<'_, anyhow::Result<String>> {
        match &self.embedder {
            Some(embedder) if embedder.servers.iter().any(|server| *server == call.server) => {
                embedder.executor.execute(call)
            }
            _ => self.core.execute(call),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use codex_mcp::McpRuntime;
    use serde_json::Map;

    use super::*;

    struct Recorder(Mutex<Vec<String>>);

    impl HookMcpExecutor for Recorder {
        fn execute(&self, call: HookMcpCall) -> BoxFuture<'_, anyhow::Result<String>> {
            self.0.lock().unwrap().push(call.server);
            async { Ok("embedder".to_string()) }.boxed()
        }
    }

    fn call(server: &str) -> HookMcpCall {
        HookMcpCall {
            server: server.to_string(),
            tool: "PreToolUse".to_string(),
            environment_id: None,
            metadata: None,
            input: Map::new(),
            timeout: Duration::from_secs(1),
        }
    }

    #[tokio::test]
    async fn a_claimed_call_reaches_the_embedder_and_another_reaches_core() {
        let recorder = Arc::new(Recorder(Mutex::new(Vec::new())));
        let routing = RoutingHookMcpExecutor {
            embedder: Some(EmbedderHookMcpExecutor {
                servers: vec!["embedder-hooks".to_string()],
                executor: recorder.clone(),
            }),
            core: CoreHookMcpExecutor {
                runtime: Arc::new(McpRuntime::empty(/*prefix_mcp_tool_names*/ false)),
                thread_id: ThreadId::new(),
            },
        };

        let claimed = routing.execute(call("embedder-hooks")).await.unwrap();
        assert_eq!(claimed, "embedder");
        // Core's runtime has no servers: the other call fails there, never reaching the embedder.
        assert!(routing.execute(call("other")).await.is_err());
        assert_eq!(
            *recorder.0.lock().unwrap(),
            vec!["embedder-hooks".to_string()]
        );
    }
}
