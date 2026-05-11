const http = require('http');
const https = require('https');

const TARGET_BASE = 'https://api.xiaomimimo.com';
const API_KEY = process.env.OPENAI_API_KEY;

function convertResponsesToChat(body) {
  const messages = [];
  if (body.instructions) {
    messages.push({ role: 'system', content: body.instructions });
  }
  if (body.input) {
    if (typeof body.input === 'string') {
      messages.push({ role: 'user', content: body.input });
    } else if (Array.isArray(body.input)) {
      for (const item of body.input) {
        if (item.type === 'message' && item.role && item.content) {
          if (typeof item.content === 'string') {
            messages.push({ role: item.role, content: item.content });
          } else if (Array.isArray(item.content)) {
            const text = item.content.map(c => c.text || '').join('');
            messages.push({ role: item.role, content: text });
          }
        }
      }
    }
  }
  return {
    model: body.model,
    messages,
    max_tokens: body.max_output_tokens || 4096,
    stream: false
  };
}

function chatToResponse(chatResponse) {
  const msg = chatResponse.choices?.[0]?.message || {};
  const content = msg.content || msg.reasoning_content || '';
  const respId = `resp_${chatResponse.id}`;
  const msgId = `msg_${chatResponse.id}`;
  return {
    id: respId,
    object: 'response',
    created_at: chatResponse.created,
    model: chatResponse.model,
    status: 'completed',
    output: [{
      id: msgId,
      type: 'message',
      status: 'completed',
      role: 'assistant',
      content: [{ type: 'output_text', text: content, annotations: [] }]
    }],
    usage: {
      input_tokens: chatResponse.usage?.prompt_tokens || 0,
      output_tokens: chatResponse.usage?.completion_tokens || 0,
      total_tokens: chatResponse.usage?.total_tokens || 0
    }
  };
}

const server = http.createServer((req, res) => {
  let body = '';
  req.on('data', chunk => body += chunk);
  req.on('end', () => {
    if (req.url === '/v1/responses') {
      try {
        const parsed = JSON.parse(body);
        const chatBody = convertResponsesToChat(parsed);
        const chatBodyStr = JSON.stringify(chatBody);

        const url = new URL('/v1/chat/completions', TARGET_BASE);
        const options = {
          hostname: url.hostname,
          port: 443,
          path: url.pathname,
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${API_KEY}`,
            'Content-Length': Buffer.byteLength(chatBodyStr)
          }
        };

        const proxyReq = https.request(options, proxyRes => {
          let data = '';
          proxyRes.on('data', chunk => data += chunk);
          proxyRes.on('end', () => {
            try {
              const chatResponse = JSON.parse(data);
              const fullResp = chatToResponse(chatResponse);
              const msg = fullResp.output[0];
              const content = msg.content[0].text;

              if (parsed.stream) {
                res.writeHead(200, {
                  'Content-Type': 'text/event-stream',
                  'Cache-Control': 'no-cache',
                  'Connection': 'keep-alive'
                });

                let seq = 0;
                const sendSSE = (eventType, data) => {
                  data.sequence_number = seq++;
                  res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
                };

                // response.created
                sendSSE('response.created', {
                  type: 'response.created',
                  response: {
                    id: fullResp.id,
                    object: 'response',
                    created_at: fullResp.created_at,
                    model: fullResp.model,
                    status: 'in_progress',
                    output: [],
                    usage: null
                  }
                });

                // response.in_progress
                sendSSE('response.in_progress', {
                  type: 'response.in_progress',
                  response: {
                    id: fullResp.id,
                    object: 'response',
                    created_at: fullResp.created_at,
                    model: fullResp.model,
                    status: 'in_progress',
                    output: [],
                    usage: null
                  }
                });

                // response.output_item.added
                sendSSE('response.output_item.added', {
                  type: 'response.output_item.added',
                  output_index: 0,
                  item: {
                    id: msg.id,
                    type: 'message',
                    status: 'in_progress',
                    role: 'assistant',
                    content: []
                  }
                });

                // response.content_part.added
                sendSSE('response.content_part.added', {
                  type: 'response.content_part.added',
                  output_index: 0,
                  content_index: 0,
                  part: { type: 'output_text', text: '', annotations: [] }
                });

                // response.output_text.delta (send whole content at once)
                sendSSE('response.output_text.delta', {
                  type: 'response.output_text.delta',
                  item_id: msg.id,
                  output_index: 0,
                  content_index: 0,
                  delta: content
                });

                // response.output_text.done
                sendSSE('response.output_text.done', {
                  type: 'response.output_text.done',
                  item_id: msg.id,
                  output_index: 0,
                  content_index: 0,
                  text: content
                });

                // response.content_part.done
                sendSSE('response.content_part.done', {
                  type: 'response.content_part.done',
                  output_index: 0,
                  content_index: 0,
                  part: { type: 'output_text', text: content, annotations: [] }
                });

                // response.output_item.done
                sendSSE('response.output_item.done', {
                  type: 'response.output_item.done',
                  output_index: 0,
                  item: msg
                });

                // response.completed
                sendSSE('response.completed', {
                  type: 'response.completed',
                  response: fullResp
                });

                res.end();
              } else {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify(fullResp));
              }
            } catch (e) {
              console.error('Error:', e.message);
              res.writeHead(500);
              res.end(JSON.stringify({ error: e.message }));
            }
          });
        });

        proxyReq.on('error', e => {
          res.writeHead(500);
          res.end(JSON.stringify({ error: e.message }));
        });

        proxyReq.write(chatBodyStr);
        proxyReq.end();
      } catch (e) {
        res.writeHead(400);
        res.end(JSON.stringify({ error: e.message }));
      }
    } else if (req.url === '/v1/models') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        object: 'list',
        data: [{ id: 'mimo-v2.5-pro', object: 'model', owned_by: 'xiaomi' }]
      }));
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
  });
});

server.listen(8765, () => {
  console.log('Codex proxy running on http://localhost:8765');
});
