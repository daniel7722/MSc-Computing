package Q2;

import Q2.HttpRequest.Method;
import java.util.List;

public class HttpRequestBuilder{

  private String url = null;
  private Method method = null;
  private List<String> params = null;
  private List<String> headers = null;
  private String body = null;

  public HttpRequestBuilder withUrl(String url) {
    this.url = url;
    return this;
  }

  public HttpRequestBuilder withMethod(Method method) {
    this.method = method;
    return this;
  }

  public HttpRequestBuilder withParams(List<String> params) {
    this.params = params;
    return this;
  }

  public HttpRequestBuilder withHeaders(List<String> headers) {
    this.headers = headers;
    return this;
  }

  public HttpRequestBuilder withBody(String body) {
    this.body = body;
    return this;
  }

  public HttpRequest build() {
    return new HttpRequest(url, method, params, headers, body);
  }
}