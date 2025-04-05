import markdown
import markdown.extensions.fenced_code
import markdown.extensions.codehilite
import markdown.extensions.tables
import markdown.extensions.toc
import markdown.extensions.attr_list
import markdown.extensions.smarty


def convert_markdown_to_html(content):
    """Set the HTML style for the content label with GitHub-style markdown rendering and LaTeX support"""
    extension_configs = {
        "codehilite": {
            "linenums": False,
            "guess_lang": True,
            "use_pygments": True,
            "css_class": "highlight",
        },
        "fenced_code": {"lang_prefix": "language-"},
    }

    # Convert markdown to HTML with extensions
    html_content = markdown.markdown(
        content,
        extensions=[
            "fenced_code",
            "codehilite",
            "tables",
            "toc",
            "attr_list",
            "smarty",
        ],
        extension_configs=extension_configs,
    )

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GitHub-Style Markdown Renderer with LaTeX</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/github.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
        
        <!-- MathJax for LaTeX rendering -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>

        <style>
            :root {{
                --primary: #60A5FA;
                --background: #FFFFFF;
                --background-secondary: #F9F9F9;
                --background-hover: #DBDBDB;
                --border: #E5E5E5;
                --text: #2C2C2E;
                --highlight-text: #2196F3;
                --success: #30D158;
                --warning: #FF9F0A;
                --error: #FF453A;
                --font-size-normal: 13px;
                --font-size-h1: 24px;
                --font-size-h2: 20px;
                --font-size-h3: 16px;
                --font-size-h4: 14px;
                --font-size-code: 12px;
            }}

            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}

            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                font-size: var(--font-size-normal);
                line-height: 1.5;
                color: var(--text);
                background-color: var(--background);
                max-width: 99%;
                margin: 0;
                padding: 0;
            }}

            .markdown-body {{
                padding: 5px;
                border: none;
                border-radius: 0px;
                background-color: var(--background);
                scrollbar-width: none; /* Firefox */
                -ms-overflow-style: none; /* IE and Edge */
            }}

            .markdown-body::-webkit-scrollbar {{
                display: none; /* Chrome, Safari, and Opera */
            }}
            
            /* Typography */
            .markdown-body h1, 
            .markdown-body h2, 
            .markdown-body h3, 
            .markdown-body h4, 
            .markdown-body h5, 
            .markdown-body h6 {{
                margin-top: 24px;
                margin-bottom: 16px;
                font-weight: 600;
                line-height: 1.25;
            }}

            .markdown-body h1 {{
                font-size: var(--font-size-h1);
                padding-bottom: 0.3em;
            }}

            .markdown-body h2 {{
                font-size: var(--font-size-h2);
                padding-bottom: 0.3em;
            }}

            .markdown-body h3 {{
                font-size: var(--font-size-h3);
            }}

            .markdown-body h4 {{
                font-size: var(--font-size-h4);
            }}

            .markdown-body p {{
                margin-bottom: 16px;
            }}

            .markdown-body a {{
                color: var(--highlight-text);
                text-decoration: none;
            }}

            .markdown-body a:hover {{
                text-decoration: underline;
            }}

            /* Lists */
            .markdown-body ul, 
            .markdown-body ol {{
                padding-left: 2em;
                margin-bottom: 16px;
            }}

            .markdown-body li {{
                margin-bottom: 4px;
            }}

            .markdown-body li + li {{
                margin-top: 4px;
            }}

            /* Blockquotes */
            .markdown-body blockquote {{
                padding: 0 1em;
                color: #6a737d;
                border-left: 0.25em solid var(--border);
                margin-bottom: 16px;
            }}

            /* Tables */
            .markdown-body table {{
                border-collapse: collapse;
                margin-bottom: 16px;
                width: auto;
                overflow: auto;
            }}

            .markdown-body table th,
            .markdown-body table td {{
                padding: 6px 13px;
                border: 1px solid var(--border);
            }}

            .markdown-body table tr {{
                background-color: var(--background);
                border-top: 1px solid var(--border);
            }}

            .markdown-body table tr:nth-child(2n) {{
                background-color: var(--background-secondary);
            }}

            .markdown-body table th {{
                font-weight: 600;
                background-color: var(--background-secondary);
            }}

            /* Code Blocks */
            .markdown-body pre {{
                position: relative;
                background-color: var(--background-secondary);
                border-radius: 6px;
                margin-bottom: 16px;
                overflow: auto;
            }}

            .markdown-body pre code {{
                display: block;
                padding: 16px;
                overflow-x: auto;
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                font-size: var(--font-size-code);
                line-height: 1.45;
                background-color: transparent;
                border-radius: 0;
            }}

            .markdown-body code {{
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
                font-size: var(--font-size-code);
                padding: 0.2em 0.4em;
                margin: 0;
                background-color: rgba(27, 31, 35, 0.05);
                border-radius: 3px;
            }}

            /* Copy Button */
            .copy-button {{
                position: absolute;
                top: 8px;
                right: 8px;
                padding: 4px 8px;
                font-size: 12px;
                color: #666;
                background-color: var(--background);
                border: 1px solid var(--border);
                border-radius: 4px;
                cursor: pointer;
                opacity: 0;
                transition: opacity 0.2s;
            }}

            .markdown-body pre:hover .copy-button {{
                opacity: 1;
            }}

            .copy-button:hover {{
                background-color: var(--background-hover);
            }}

            .copy-button.copied {{
                color: var(--success);
                border-color: var(--success);
            }}

            /* Status colors */
            .success {{
                color: var(--success);
            }}

            .warning {{
                color: var(--warning);
            }}

            .error {{
                color: var(--error);
            }}

            /* TOC */
            .toc {{
                background-color: var(--background-secondary);
                border: 1px solid var(--border);
                border-radius: 6px;
                padding: 16px;
                margin-bottom: 16px;
            }}

            .toc-title {{
                font-weight: 600;
                margin-bottom: 8px;
            }}

            /* GitHub-style horizontal rule */
            .markdown-body hr {{
                display: none;
            }}
            
            /* Styles for LaTeX blocks */
            .math-block {{
                overflow-x: auto;
                margin: 16px 0;
                padding: 8px;
                background-color: var(--background-secondary);
                border-radius: 6px;
            }}
        </style>
    </head>
    <body>
        <div class="markdown-body" id="content">
            {html_content}
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                hljs.configure({{
                    ignoreUnescapedHTML: true
                }});

                // Add copy buttons to code blocks
                document.querySelectorAll('pre').forEach(function(codeBlock) {{
                    if (!codeBlock.querySelector('.copy-button')) {{
                        const copyButton = document.createElement('button');
                        copyButton.className = 'copy-button';
                        copyButton.textContent = 'Copy';
                        codeBlock.appendChild(copyButton);

                        copyButton.addEventListener('click', function() {{
                            const code = codeBlock.querySelector('code');
                            const range = document.createRange();
                            range.selectNode(code);
                            window.getSelection().removeAllRanges();
                            window.getSelection().addRange(range);
                            
                            try {{
                                document.execCommand('copy');
                                copyButton.textContent = 'Copied!';
                                copyButton.classList.add('copied');
                                
                                setTimeout(function() {{
                                    copyButton.textContent = 'Copy';
                                    copyButton.classList.remove('copied');
                                }}, 2000);
                            }} catch (err) {{
                                console.error('Failed to copy: ', err);
                                copyButton.textContent = 'Failed';
                            }}
                            
                            window.getSelection().removeAllRanges();
                        }});
                    }}
                }});

                // Initialize syntax highlighting
                document.querySelectorAll('pre code').forEach((el) => {{
                    hljs.highlightElement(el);
                }});
                
                // Configure MathJax
                window.MathJax = {{
                    tex: {{
                        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                        processEscapes: true,
                        tags: 'ams'
                    }},
                    options: {{
                        enableMenu: false,
                        renderActions: {{
                            addMenu: [0, '', '']
                        }}
                    }},
                    startup: {{
                        pageReady: () => {{
                            return MathJax.startup.defaultPageReady().then(() => {{
                                // Wrap display math in styled divs
                                document.querySelectorAll('.MathJax_Display').forEach(element => {{
                                    const wrapper = document.createElement('div');
                                    wrapper.className = 'math-block';
                                    element.parentNode.insertBefore(wrapper, element);
                                    wrapper.appendChild(element);
                                }});
                            }});
                        }}
                    }}
                }};
            }});
        </script>
    </body>
    </html>
    """
